"""
    confirm_routed_multilead.jl

Robustness confirmation for the capstone result (TEST AUROC 0.851 on one split).
The winning hyperparameters are FIXED; only the data split (seed) and the lead set
vary. This isolates split variance from search variance — the honest question is
"does this configuration generalize, or did it draw a lucky test split?".

For each SEED (fresh val/test split) and each lead set (6-lead vs all-12) it:
  - extracts the routed multi-lead [output;hidden] readout for val + test recordings,
  - fits the diagonal LDA on validation, scores the held-out test once,
  - reports test AUROC (+ bootstrap CI), balanced accuracy and MCC.
Then it summarises mean ± std across seeds per lead set.
"""

include("./modules/Layers.jl")
include("./modules/Signals.jl")
include("./modules/Registry.jl")
include("./modules/Metrics.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils,
      .Registry, .Metrics
using Random, ProgressMeter, Statistics
using StatsBase: sample

# ----- Winning params from paramsearch_routed_multilead.jl -----
const Δ            = 0.12970001789262522
const pulse_amp    = 170.48121762522024
const ltp_rate     = 0.18935145456196095
const ltp_rate_out = 0.08378635445090134
const τ_s          = 42.831648889100016
const τ_s_output   = 25.003717150125617
const τ_pretrace   = 34.510086127922726
const τ_posttrace  = 47.00571781147488
const R_m_input    = 1.0674163386846072
const R_m_output   = 2.8737140296290087
const τ_m_input    = 9.41794217755773
const τ_m_output   = 2.4038829712094394
const τ_ref_input  = 7.209849285084017
const τ_ref_output = 0.5566821677223849
const N_in         = 20
const N_hid        = 25
const N_out        = 25
const density      = 0.7793537471176322
const inhib_den    = 0.2926531513089269
const inhib_str    = 0.05207908906117163

const dt           = 0.001
const READOUT_TIME = 12.0
const SETTLE       = 5.0
const SAMPLE_INT   = 0.2
const FS           = 1000.0
const PRE_R        = 0.25
const GAP          = 100.0
const R_IDX        = round(Int, PRE_R * FS) + 1

# ----- Confirmation config -----
const SEEDS          = [11, 22, 33]            # fresh val/test splits
const VAL_PER_CLASS  = 20
const TEST_PER_CLASS = 40
const N_BOOT         = 1000
const LDA_SHRINK     = 1e-3
const LEADSETS = ("6-lead" => [2,3,6,8,9,10], "12-lead" => collect(1:12))

seg_base(t) = t < R_IDX - 60 ? 0 : (t <= R_IDX + 80 ? 2 : 4)
const NET_RNG_BASE = 20240  # fixed network-init seed base (config is fixed)

function build_routed()
    inp(rev) = Neuron(rev ? "id" : "iu"; R_m=R_m_input, τ_m=τ_m_input, τ_s=τ_s, τ_ref=τ_ref_input, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace, isreverse=rev)
    hidt = Neuron("h"; R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    outt = Neuron("o"; R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    exc = Synapse(1,2; learningrate=ltp_rate, wmax=1.0); exc_out = Synapse(1,2; learningrate=ltp_rate_out, wmax=1.0)
    inh = Synapse(1,1; learningrate=0.0, wmax=1.0, isinhibitory=true)
    return inp, hidt, outt, exc, exc_out, inh
end

function routed_net(rng)
    inp, hidt, outt, exc, exc_out, inh = build_routed()
    ml(t,n,nm) = NeuronLayer(n, t; name=nm, V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15, rng=rng)
    ins = [ml(inp(iseven(i)), N_in, "in$(i)") for i in 1:6]
    hd = ml(hidt, N_hid, "h"); ou = ml(outt, N_out, "o")
    syns = SynapseLayer[]
    for i in 1:6
        push!(syns, SynapseLayer(ins[i], hd, exc; dist=NormalDist(0.5,0.2), density=density, pre_idx=i, post_idx=7, rng=rng))
    end
    push!(syns, SynapseLayer(hd, ou, exc_out; dist=NormalDist(0.5,0.2), density=density, pre_idx=7, post_idx=8, rng=rng))
    idist = UniformDist(0.05, inhib_str)
    for i in 1:6; push!(syns, SynapseLayer(ins[i], ins[i], inh; dist=idist, density=inhib_den, pre_idx=i, post_idx=i, rng=rng)); end
    push!(syns, SynapseLayer(hd, hd, inh; dist=idist, density=inhib_den, pre_idx=7, post_idx=7, rng=rng))
    push!(syns, SynapseLayer(ou, ou, inh; dist=idist, density=inhib_den, pre_idx=8, post_idx=8, rng=rng))
    return LayeredNetwork(vcat(ins, [hd, ou]), syns)
end

function delta_mod(beat)
    n=length(beat); lvl=beat[1]; out=Spike[]
    for t in 2:n
        d=beat[t]-lvl
        if d >= Δ; push!(out, Spike(Float64(t), true, "d")); lvl+=Δ
        elseif d <= -Δ; push!(out, Spike(Float64(t), false, "d")); lvl-=Δ; end
    end
    return out
end

function routed_pulses(rec, lead, nsteps)
    filt = get_filtered_signal(load_raw_signal(rec.patient, rec.session; lead=lead))
    beats = segment_beats(filt, get_R_peaks(filt; fs=FS); fs=FS)
    arr = [zeros(nsteps) for _ in 1:6]; offset = 0.0
    for beat in beats
        for s in delta_mod(normalize_beat(beat))
            step = Int(floor(s.time + offset)) + 1
            1 <= step <= nsteps || continue
            base = seg_base(s.time)
            s.polarity ? (arr[base+1][step] += pulse_amp) : (arr[base+2][step] -= pulse_amp)
        end
        offset += length(beat) + GAP
    end
    return arr
end

function feature(rec, leads)
    ns = Int(round(READOUT_TIME / dt)); out = Float64[]
    for lead in leads
        arr = routed_pulses(rec, lead, ns)
        net = routed_net(MersenneTwister(NET_RNG_BASE + lead))
        input_fn = t -> begin
            idx = clamp(Int(floor(t/dt)) + 1, 1, ns)
            v = zeros(length(net.neuronlayers)); @inbounds for k in 1:6; v[k]=arr[k][idx]; end; v
        end
        po=zeros(Int,N_out); ph=zeros(Int,N_hid); npost=0; last_t=-1.0; nxt=0.0
        cb = function(t, net, step)
            if t >= nxt - 1e-9
                if t > SETTLE
                    po .+= (net.neuronlayers[8].t_lastout .> last_t)
                    ph .+= (net.neuronlayers[7].t_lastout .> last_t)
                    npost += 1
                end
                last_t = t; nxt += SAMPLE_INT
            end
        end
        runlayers!(net, dt, READOUT_TIME; inputfn=input_fn, callback=cb)
        npost = max(npost,1); append!(out, po./npost); append!(out, ph./npost)
    end
    return out
end

function fit_lda(X0, X1)
    μ0 = vec(mean(X0, dims=1)); μ1 = vec(mean(X1, dims=1))
    n0, n1 = size(X0,1), size(X1,1)
    vp = ((n0-1).*vec(var(X0,dims=1)) .+ (n1-1).*vec(var(X1,dims=1))) ./ max(1, n0+n1-2) .+ LDA_SHRINK
    return (w=(μ1.-μ0)./vp, mid=(μ0.+μ1)./2)
end
score_lda(m,x) = sum(m.w.*(x.-m.mid))
mat(V) = (d=length(V[1]); reduce(vcat,[reshape(v,1,d) for v in V]))

function boot_ci(sc, lab; nb=N_BOOT, rng=MersenneTwister(0))
    n=length(sc); st=Float64[]
    for _ in 1:nb
        idx=rand(rng,1:n,n); s=sc[idx]; l=lab[idx]
        (count(l)<1||count(!,l)<1) && continue
        a=auroc(s,l); isnan(a)||push!(st,a)
    end
    sort!(st)
    return (isempty(st) ? NaN : st[max(1,floor(Int,0.025*length(st)))],
            isempty(st) ? NaN : st[max(1,ceil(Int,0.975*length(st)))])
end

function eval_split(seed, leads)
    lab0 = filter(r -> r.label != :unknown, build_registry("./ecg-db"))
    hall = filter(r -> r.label == :healthy, lab0); iall = filter(r -> r.label == :infarction, lab0)
    hsh = hall[randperm(MersenneTwister(seed), length(hall))]
    ish = iall[randperm(MersenneTwister(seed+1), length(iall))]
    test_h=hsh[1:TEST_PER_CLASS]; test_i=ish[1:TEST_PER_CLASS]
    val_h=hsh[TEST_PER_CLASS+1:TEST_PER_CLASS+VAL_PER_CLASS]; val_i=ish[TEST_PER_CLASS+1:TEST_PER_CLASS+VAL_PER_CLASS]
    Hv=mat([feature(r,leads) for r in val_h]); Iv=mat([feature(r,leads) for r in val_i])
    m=fit_lda(Hv,Iv)
    sc=Float64[]; lab=Bool[]
    for r in test_h; push!(sc, score_lda(m, feature(r,leads))); push!(lab,false); end
    for r in test_i; push!(sc, score_lda(m, feature(r,leads))); push!(lab,true);  end
    a=auroc(sc,lab); lo,hi=boot_ci(sc,lab); cc=confusion_counts(lab, sc .>= 0.0)
    return (auc=a, lo=lo, hi=hi, ba=balanced_acc(cc), mcc=mcc(cc))
end

# -----------------------------------------------------------------------------------
println("="^78)
println("ROBUSTNESS CONFIRMATION — fixed winning config, varying split & lead set")
println("="^78)
println("  Seeds: $(SEEDS) | Val $(VAL_PER_CLASS)/class | Test $(TEST_PER_CLASS)/class")

summary = Dict{String,Vector{Float64}}()
for (name, leads) in LEADSETS
    println("\n", "-"^78, "\n$name  ($(length(leads)) leads: $(leads))\n", "-"^78)
    println("  seed   AUROC   95% CI            bal.acc   MCC")
    aucs = Float64[]
    @showprogress for s in SEEDS
        r = eval_split(s, leads)
        push!(aucs, r.auc)
        println("  $(rpad(s,5)) $(rpad(round(r.auc,digits=3),7)) [$(round(r.lo,digits=3)), $(round(r.hi,digits=3))]   " *
                "$(rpad(string(round(100*r.ba,digits=1))*"%",8)) $(round(r.mcc,digits=3))")
    end
    summary[name] = aucs
    println("  → $name mean AUROC = $(round(mean(aucs),digits=3)) ± $(round(std(aucs),digits=3))")
end

println("\n", "="^78)
println("SUMMARY")
println("="^78)
for (name, _) in LEADSETS
    a = summary[name]
    println("  $(rpad(name,8)): test AUROC $(round(mean(a),digits=3)) ± $(round(std(a),digits=3))  (per-seed: $(join(round.(a,digits=3), ", ")))")
end
six = summary["6-lead"]; twelve = summary["12-lead"]
d = mean(twelve) - mean(six)
println("\n  12-lead − 6-lead mean AUROC: $(round(d,digits=3))")
if abs(d) < 0.02
    println("  → 6 leads suffice: all-12 gives no meaningful gain (wearable-friendly result).")
elseif d >= 0.02
    println("  → All 12 leads help: report the 12-lead number as the headline.")
else
    println("  → 6 leads slightly better (12-lead may add noise from low-information leads).")
end
if std(six) <= 0.04 && std(twelve) <= 0.04
    println("  → Stable across splits (std ≤ 0.04): the ~0.85 result is robust")
else
    println("  → Some split sensitivity (std > 0.04): report mean ± std")
end
println("\n", "="^78, "\nDONE\n", "="^78)
