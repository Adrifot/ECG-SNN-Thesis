"""
    eval_routed_multilead.jl

Three conditions on the same recordings:
  baseline              plain 2-input network, lead II            
  routed                P/QRS/T input groups, lead II            
  routed + multi-lead   routed network across all leads           

Readout: per-output-neuron firing-rate vector -> cross-validated diagonal LDA.
"""

include("../modules/Layers.jl")
include("../modules/Signals.jl")
include("../modules/Registry.jl")
include("../modules/Metrics.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils,
      .Registry, .Metrics
using Statistics, Random, ProgressMeter
using StatsBase: sample

# ----- Best params  -----
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
const SEED         = 42
const N_PER        = 30         
const KFOLD        = 5
const LDA_SHRINK   = 1e-3
const FS           = 1000.0
const PRE_R        = 0.25
const POST_R       = 0.45
const GAP          = 100.0
const R_IDX        = round(Int, PRE_R * FS) + 1
const LEADS        = 1:12         # 12 leads (i..v6); .xyz leads excluded

# Within-beat windows (sample indices) → P, QRS, T (T = repolarisation, ST+T)
seg_of(t) = t < R_IDX - 60 ? :P : (t <= R_IDX + 80 ? :QRS : :T)

# ----- Neuron templates -----
inp_t(rev) = Neuron(rev ? "in_down" : "in_up"; R_m=R_m_input, τ_m=τ_m_input, τ_s=τ_s,
                    τ_ref=τ_ref_input, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace, isreverse=rev)
hid_t() = Neuron("hidden"; R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
out_t() = Neuron("output"; R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
mklayer(t, n, nm) = NeuronLayer(n, t; name=nm, V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
inhibify(l, i) = SynapseLayer(l, l, Synapse(1,1; learningrate=0.0, wmax=1.0, isinhibitory=true);
                              dist=UniformDist(0.05,inhib_str), density=inhib_den, pre_idx=i, post_idx=i)

# Plain: (up, down) → hidden → output 
function build_plain()
    Random.seed!(SEED)
    exc = Synapse(1,2; learningrate=ltp_rate, wmax=1.0); exc_out = Synapse(1,2; learningrate=ltp_rate_out, wmax=1.0)
    up=mklayer(inp_t(false), N_in, "up"); dn=mklayer(inp_t(true), N_in, "down"); hd=mklayer(hid_t(), N_hid, "h"); ou=mklayer(out_t(), N_out, "o")
    syns = [
        SynapseLayer(up, hd, exc; dist=NormalDist(0.5,0.2), density=density, pre_idx=1, post_idx=3),
        SynapseLayer(dn, hd, exc; dist=NormalDist(0.5,0.2), density=density, pre_idx=2, post_idx=3),
        SynapseLayer(hd, ou, exc_out; dist=NormalDist(0.5,0.2), density=density, pre_idx=3, post_idx=4),
        inhibify(up,1), inhibify(dn,2), inhibify(hd,3), inhibify(ou,4),
    ]
    return LayeredNetwork([up,dn,hd,ou], syns), 4
end

# Routed: (P_up,P_dn,QRS_up,QRS_dn,T_up,T_dn) → hidden → output 
function build_routed()
    Random.seed!(SEED)
    exc = Synapse(1,2; learningrate=ltp_rate, wmax=1.0); exc_out = Synapse(1,2; learningrate=ltp_rate_out, wmax=1.0)
    ins = [mklayer(inp_t(isodd(i) ? false : true), N_in, "in$(i)") for i in 1:6]  # odd=up, even=down
    hd = mklayer(hid_t(), N_hid, "h"); ou = mklayer(out_t(), N_out, "o")
    layers = vcat(ins, [hd, ou])    # indices: inputs 1..6, hidden 7, output 8
    syns = SynapseLayer[]
    for i in 1:6
        push!(syns, SynapseLayer(ins[i], hd, exc; dist=NormalDist(0.5,0.2), density=density, pre_idx=i, post_idx=7))
    end
    push!(syns, SynapseLayer(hd, ou, exc_out; dist=NormalDist(0.5,0.2), density=density, pre_idx=7, post_idx=8))
    for i in 1:6; push!(syns, inhibify(ins[i], i)); end
    push!(syns, inhibify(hd,7)); push!(syns, inhibify(ou,8))
    return LayeredNetwork(layers, syns), 8
end

# ----- Local delta modulation -----
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
    raw = load_raw_signal(rec.patient, rec.session; lead=lead)
    filt = get_filtered_signal(raw)
    peaks = get_R_peaks(filt; fs=FS)
    beats = segment_beats(filt, peaks; fs=FS)
    arr = [zeros(nsteps) for _ in 1:6]   # P_up,P_dn,QRS_up,QRS_dn,T_up,T_dn
    seg_idx = Dict(:P=>0, :QRS=>2, :T=>4)
    offset = 0.0
    for beat in beats
        for s in delta_mod(normalize_beat(beat))
            gt = s.time + offset
            step = Int(floor(gt)) + 1
            1 <= step <= nsteps || continue
            base = seg_idx[seg_of(s.time)]
            if s.polarity; arr[base+1][step] += pulse_amp
            else;          arr[base+2][step] -= pulse_amp; end
        end
        offset += length(beat) + GAP
    end
    return arr
end

# Plain drive: collapse the 6 routed arrays to total up/down
function plain_pulses(rec, lead, nsteps)
    a = routed_pulses(rec, lead, nsteps)
    return [a[1].+a[3].+a[5], a[2].+a[4].+a[6]]   # up = ΣP/QRS/T up, down = Σ down
end

# ----- Simulate - return per-output-neuron post-settle firing-rate vector -----
function run_readout(net, out_idx, pulse_arrays)
    nsteps = length(pulse_arrays[1])
    input_fn = t -> begin
        idx = clamp(Int(floor(t/dt)) + 1, 1, nsteps)
        v = zeros(length(net.neuronlayers))
        @inbounds for k in 1:length(pulse_arrays); v[k] = pulse_arrays[k][idx]; end
        return v
    end
    nout = net.neuronlayers[out_idx].N
    per = zeros(Int, nout); n_post = 0; last_t = -1.0; next_s = 0.0
    cb = function(t, net, step)
        if t >= next_s - 1e-9
            t > SETTLE && (per .+= (net.neuronlayers[out_idx].t_lastout .> last_t); n_post += 1)
            last_t = t; next_s += SAMPLE_INT
        end
    end
    runlayers!(net, dt, READOUT_TIME; inputfn=input_fn, callback=cb)
    return per ./ max(n_post, 1)
end

nsteps_const() = Int(round(READOUT_TIME / dt))

feat_plain(rec)  = run_readout(build_plain()...,  plain_pulses(rec, 2, nsteps_const()))
feat_routed(rec) = run_readout(build_routed()..., routed_pulses(rec, 2, nsteps_const()))
function feat_routed_ml(rec)
    ns = nsteps_const()
    reduce(vcat, [run_readout(build_routed()..., routed_pulses(rec, ld, ns)) for ld in LEADS])
end

# ----- diagonal LDA + stratified CV -----
function fit_diag_lda(X0, X1)
    μ0=vec(mean(X0,dims=1)); μ1=vec(mean(X1,dims=1)); n0,n1=size(X0,1),size(X1,1)
    vpool=((n0-1).*vec(var(X0,dims=1)).+(n1-1).*vec(var(X1,dims=1)))./max(1,n0+n1-2).+LDA_SHRINK
    return (w=(μ1.-μ0)./vpool, mid=(μ0.+μ1)./2)
end
lda_score(m,x)=sum(m.w.*(x.-m.mid))
function cv_lda(X0, X1)
    n0,n1=size(X0,1),size(X1,1); rng=MersenneTwister(SEED)
    f0=[Int[] for _ in 1:KFOLD]; f1=[Int[] for _ in 1:KFOLD]
    for (i,idx) in enumerate(randperm(rng,n0)); push!(f0[mod1(i,KFOLD)],idx); end
    for (i,idx) in enumerate(randperm(rng,n1)); push!(f1[mod1(i,KFOLD)],idx); end
    sc=Float64[]; lab=Bool[]
    for k in 1:KFOLD
        te0=f0[k]; te1=f1[k]; tr0=setdiff(1:n0,te0); tr1=setdiff(1:n1,te1)
        (isempty(tr0)||isempty(tr1)||isempty(te0)||isempty(te1)) && continue
        m=fit_diag_lda(X0[tr0,:],X1[tr1,:])
        for i in te0; push!(sc,lda_score(m,X0[i,:])); push!(lab,false); end
        for i in te1; push!(sc,lda_score(m,X1[i,:])); push!(lab,true);  end
    end
    a=auroc(sc,lab); cc=confusion_counts(lab, sc .>= 0.0)
    return a, balanced_acc(cc), mcc(cc)
end

function condition(name, featfn, h_recs, i_recs)
    println("  $name…")
    H = @showprogress [featfn(r) for r in h_recs]
    I = @showprogress [featfn(r) for r in i_recs]
    d = length(H[1])
    X0 = reduce(vcat,[reshape(v,1,d) for v in H]); X1 = reduce(vcat,[reshape(v,1,d) for v in I])
    a, ba, mc = cv_lda(X0, X1)
    return (auc=a, ba=ba, mcc=mc, d=d)
end

# --------------------------------------------------------------------------------------------------
Random.seed!(SEED)
labelled = filter(r -> r.label != :unknown, build_registry("./ecg-db"))
healthy = filter(r -> r.label == :healthy, labelled)
infarction = filter(r -> r.label == :infarction, labelled)
h_recs = [sample(healthy) for _ in 1:N_PER]
i_recs = [sample(infarction) for _ in 1:N_PER]

println("="^78)
println("SEGMENT ROUTING × MULTI-LEAD — $(N_PER) healthy vs $(N_PER) infarction (CV LDA)")
println("="^78)
rb = condition("baseline (plain, lead II)", feat_plain, h_recs, i_recs)
rr = condition("routed (P/QRS/T, lead II)", feat_routed, h_recs, i_recs)
rm = condition("routed + multi-lead ($(length(LEADS)) leads)", feat_routed_ml, h_recs, i_recs)

println("\n", "="^78)
println("RESULTS  ($(KFOLD)-fold cross-validated; feature dim in parentheses)")
println("="^78)
println("  condition                         AUROC    bal.acc    MCC      dim")
println("  " * "-"^72)
for (nm, r) in [("baseline (plain, lead II)", rb), ("routed (P/QRS/T, lead II)", rr),
                ("routed + multi-lead", rm)]
    println("  $(rpad(nm,32)) $(rpad(round(r.auc,digits=3),8)) $(rpad(string(round(100*r.ba,digits=1))*"%",10)) " *
            "$(rpad(round(r.mcc,digits=3),8)) $(r.d)")
end

println("\n", "="^78)
println("VERDICT")
println("="^78)
println("  baseline AUROC=$(round(rb.auc,digits=3))  →  routed=$(round(rr.auc,digits=3))  →  routed+ML=$(round(rm.auc,digits=3))")
gain_route = rr.auc - rb.auc
gain_lead  = rm.auc - rr.auc
println("  routing gain: $(round(gain_route,digits=3))   |   multi-lead gain: $(round(gain_lead,digits=3))")
if rm.auc >= 0.80
    println("  ✓ Strong: AUROC ≥ 0.80. The combined architecture works.")
elseif rm.auc >= rb.auc + 0.07
    println("  ✓ Meaningful improvement over the lead-II baseline.")
else
    println("  ~ Limited gain.")
end
println("\n  Provisional: cross-validated on the SAME $(N_PER)/class pool.")
println("\n", "="^78, "\nDONE\n", "="^78)
