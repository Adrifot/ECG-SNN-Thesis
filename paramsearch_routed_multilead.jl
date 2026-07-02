include("./modules/Layers.jl")
include("./modules/Signals.jl")
include("./modules/Registry.jl")
include("./modules/ClassificationMetrics.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils,
      .Registry, .ClassificationMetrics
using Random, ProgressMeter, Statistics
using StatsBase: sample

# ----- config -----
const SEARCH_ITERATIONS  = 150
const FOCUSED_ITERATIONS = 30
const FOCUSED_TOPK       = 3
const VAL_PER_CLASS      = 15     # validation recordings per class (config selection)
const TEST_PER_CLASS     = 60     # held-out test recordings per class (winner only)
const KFOLD              = 5
const N_BOOT             = 2000
const LDA_SHRINK         = 1e-3
# 6 diagnostically-spread leads: II, III, aVF (inferior) + V2, V3, V4 (anterior).
const LEADS              = [2, 3, 6, 8, 9, 10]
const dt           = 0.001
const READOUT_TIME = 12.0
const SETTLE       = 5.0
const SAMPLE_INT   = 0.2
const FS           = 1000.0
const PRE_R        = 0.25
const GAP          = 100.0
const R_IDX        = round(Int, PRE_R * FS) + 1
const SEED         = 67

seg_base(t) = t < R_IDX - 60 ? 0 : (t <= R_IDX + 80 ? 2 : 4)   # P=0, QRS=2, T=4

# ----- HyperParams (per-layer neuron counts; wider intervals) -----
struct HyperParams
    Δ::Float64; pulse_amp::Float64; ltp_rate::Float64; ltp_rate_out::Float64
    τ_s::Float64; τ_s_output::Float64; τ_pretrace::Float64; τ_posttrace::Float64
    R_m_input::Float64; R_m_output::Float64; τ_m_input::Float64; τ_m_output::Float64
    τ_ref_input::Float64; τ_ref_output::Float64
    N_in::Int; N_hid::Int; N_out::Int
    density::Float64; inhib_den::Float64; inhib_str::Float64
end

param_hash(hp, salt=0) = mod(hash((hp.Δ,hp.pulse_amp,hp.ltp_rate,hp.ltp_rate_out,hp.τ_s,hp.τ_s_output,
    hp.τ_pretrace,hp.τ_posttrace,hp.R_m_input,hp.R_m_output,hp.τ_m_input,hp.τ_m_output,hp.τ_ref_input,
    hp.τ_ref_output,hp.N_in,hp.N_hid,hp.N_out,hp.density,hp.inhib_den,hp.inhib_str)) + salt, typemax(Int32))

rnd(a,b,rng) = rand(rng)*(b-a)+a
const N_CHOICES = [10, 15, 20, 25, 30]

function sample_random(rng)
    HyperParams(
        rnd(0.04,0.25,rng), rnd(10.0,350.0,rng),         # Δ, pulse_amp 
        rnd(0.02,0.30,rng), rnd(0.02,0.30,rng),          # learning rates 
        rnd(5.0,55.0,rng), rnd(3.0,45.0,rng),            # τ_s, τ_s_output
        rnd(3.0,35.0,rng), rnd(15.0,65.0,rng),           # pre/post trace
        rnd(0.5,9.0,rng), rnd(0.3,6.0,rng),              # R_m in/out
        rnd(2.0,12.0,rng), rnd(1.0,12.0,rng),            # τ_m in/out
        rnd(0.5,9.0,rng), rnd(0.5,4.0,rng),              # τ_ref in/out
        rand(rng,N_CHOICES), rand(rng,N_CHOICES), rand(rng,N_CHOICES),
        rnd(0.5,0.95,rng), rnd(0.3,0.85,rng), rnd(0.05,0.6,rng),
    )
end

function sample_focused(rng, b; f=0.15)
    p(v,lo,hi) = clamp(v*(1+(2*rand(rng)-1)*f), lo, hi)
    jitterN(n) = N_CHOICES[clamp(findfirst(==(n),N_CHOICES) + rand(rng,-1:1), 1, length(N_CHOICES))]
    HyperParams(
        p(b.Δ,0.02,0.25), p(b.pulse_amp,5.0,360.0), p(b.ltp_rate,0.01,0.3), p(b.ltp_rate_out,0.01,0.3),
        p(b.τ_s,1.0,60.0), p(b.τ_s_output,1.0,50.0), p(b.τ_pretrace,2.0,40.0), p(b.τ_posttrace,10.0,65.0),
        p(b.R_m_input,0.1,10.0), p(b.R_m_output,0.1,8.0), p(b.τ_m_input,0.5,12.0), p(b.τ_m_output,0.5,12.0),
        p(b.τ_ref_input,0.5,10.0), p(b.τ_ref_output,0.5,4.0),
        jitterN(b.N_in), jitterN(b.N_hid), jitterN(b.N_out),
        p(b.density,0.4,0.97), p(b.inhib_den,0.2,0.9), p(b.inhib_str,0.05,0.8),
    )
end

# ----- Routed network with per-layer neuron counts -----
function build_routed(hp, rng)
    inp(rev) = Neuron(rev ? "id" : "iu"; R_m=hp.R_m_input, τ_m=hp.τ_m_input, τ_s=hp.τ_s, τ_ref=hp.τ_ref_input, τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace, isreverse=rev)
    hidt = Neuron("h"; R_m=hp.R_m_output, τ_m=hp.τ_m_output, τ_s=hp.τ_s_output, τ_ref=hp.τ_ref_output, τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace)
    outt = Neuron("o"; R_m=hp.R_m_output, τ_m=hp.τ_m_output, τ_s=hp.τ_s_output, τ_ref=hp.τ_ref_output, τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace)
    exc = Synapse(1,2; learningrate=hp.ltp_rate, wmax=1.0); exc_out = Synapse(1,2; learningrate=hp.ltp_rate_out, wmax=1.0)
    inh = Synapse(1,1; learningrate=0.0, wmax=1.0, isinhibitory=true)
    ml(t,n,nm) = NeuronLayer(n, t; name=nm, V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15, rng=rng)
    ins = [ml(inp(iseven(i)), hp.N_in, "in$(i)") for i in 1:6]   # odd=up, even=down
    hd = ml(hidt, hp.N_hid, "h"); ou = ml(outt, hp.N_out, "o")
    layers = vcat(ins, [hd, ou])
    syns = SynapseLayer[]
    for i in 1:6
        push!(syns, SynapseLayer(ins[i], hd, exc; dist=NormalDist(0.5,0.2), density=hp.density, pre_idx=i, post_idx=7, rng=rng))
    end
    push!(syns, SynapseLayer(hd, ou, exc_out; dist=NormalDist(0.5,0.2), density=hp.density, pre_idx=7, post_idx=8, rng=rng))
    idist = UniformDist(0.05, hp.inhib_str)
    for i in 1:6; push!(syns, SynapseLayer(ins[i], ins[i], inh; dist=idist, density=hp.inhib_den, pre_idx=i, post_idx=i, rng=rng)); end
    push!(syns, SynapseLayer(hd, hd, inh; dist=idist, density=hp.inhib_den, pre_idx=7, post_idx=7, rng=rng))
    push!(syns, SynapseLayer(ou, ou, inh; dist=idist, density=hp.inhib_den, pre_idx=8, post_idx=8, rng=rng))
    return LayeredNetwork(layers, syns)
end

function delta_mod(beat, Δ)
    n=length(beat); lvl=beat[1]; out=Spike[]
    for t in 2:n
        d=beat[t]-lvl
        if d >= Δ; push!(out, Spike(Float64(t), true, "d")); lvl+=Δ
        elseif d <= -Δ; push!(out, Spike(Float64(t), false, "d")); lvl-=Δ; end
    end
    return out
end

function routed_pulses(rec, lead, nsteps, hp)
    filt = get_filtered_signal(load_raw_signal(rec.patient, rec.session; lead=lead))
    beats = segment_beats(filt, get_R_peaks(filt; fs=FS); fs=FS)
    arr = [zeros(nsteps) for _ in 1:6]
    offset = 0.0
    for beat in beats
        for s in delta_mod(normalize_beat(beat), hp.Δ)
            step = Int(floor(s.time + offset)) + 1
            1 <= step <= nsteps || continue
            base = seg_base(s.time)
            s.polarity ? (arr[base+1][step] += hp.pulse_amp) : (arr[base+2][step] -= hp.pulse_amp)
        end
        offset += length(beat) + GAP
    end
    return arr
end

# Per-recording feature: concat over leads of [output-rate-vector ; hidden-rate-vector]
function feature(rec, hp)
    ns = Int(round(READOUT_TIME / dt))
    out = Float64[]
    for lead in LEADS
        arr = routed_pulses(rec, lead, ns, hp)
        net = build_routed(hp, MersenneTwister(param_hash(hp) + lead))
        input_fn = t -> begin
            idx = clamp(Int(floor(t/dt)) + 1, 1, ns)
            v = zeros(length(net.neuronlayers)); @inbounds for k in 1:6; v[k]=arr[k][idx]; end; v
        end
        po = zeros(Int, hp.N_out); ph = zeros(Int, hp.N_hid); npost=0; last_t=-1.0; nxt=0.0
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
        npost = max(npost, 1)
        append!(out, po ./ npost); append!(out, ph ./ npost)
    end
    return out
end

# ----- diagonal LDA + CV / fit-test -----
function fit_lda(X0, X1)
    μ0=vec(mean(X0,dims=1)); μ1=vec(mean(X1,dims=1)); n0,n1=size(X0,1),size(X1,1)
    vp=((n0-1).*vec(var(X0,dims=1)).+(n1-1).*vec(var(X1,dims=1)))./max(1,n0+n1-2).+LDA_SHRINK
    return (w=(μ1.-μ0)./vp, mid=(μ0.+μ1)./2)
end
score_lda(m,x)=sum(m.w.*(x.-m.mid))
mat(V) = (d=length(V[1]); reduce(vcat,[reshape(v,1,d) for v in V]))

function cv_auc(X0, X1)
    n0,n1=size(X0,1),size(X1,1); rng=MersenneTwister(SEED)
    f0=[Int[] for _ in 1:KFOLD]; f1=[Int[] for _ in 1:KFOLD]
    for (i,idx) in enumerate(randperm(rng,n0)); push!(f0[mod1(i,KFOLD)],idx); end
    for (i,idx) in enumerate(randperm(rng,n1)); push!(f1[mod1(i,KFOLD)],idx); end
    sc=Float64[]; lab=Bool[]
    for k in 1:KFOLD
        te0=f0[k]; te1=f1[k]; tr0=setdiff(1:n0,te0); tr1=setdiff(1:n1,te1)
        (isempty(tr0)||isempty(tr1)||isempty(te0)||isempty(te1)) && continue
        m=fit_lda(X0[tr0,:],X1[tr1,:])
        for i in te0; push!(sc,score_lda(m,X0[i,:])); push!(lab,false); end
        for i in te1; push!(sc,score_lda(m,X1[i,:])); push!(lab,true);  end
    end
    a=auroc(sc,lab); return isnan(a) ? 0.5 : a
end

function boot_auc_ci(scores, labels; nboot=N_BOOT, rng=MersenneTwister(SEED))
    n=length(scores); pt=auroc(scores,labels); stats=Float64[]
    for _ in 1:nboot
        idx=rand(rng,1:n,n)
        s=scores[idx]; l=labels[idx]
        (count(l)<1 || count(!,l)<1) && continue
        a=auroc(s,l); isnan(a) || push!(stats,a)
    end
    sort!(stats)
    lo = isempty(stats) ? NaN : stats[max(1,floor(Int,0.025*length(stats)))]
    hi = isempty(stats) ? NaN : stats[max(1,ceil(Int,0.975*length(stats)))]
    return pt, lo, hi
end

# ----------------------------------------------------------------------------------------------------------
Random.seed!(SEED)
labelled   = filter(r -> r.label != :unknown, build_registry("./ecg-db"))
hall = filter(r -> r.label == :healthy, labelled); iall = filter(r -> r.label == :infarction, labelled)
hsh = hall[randperm(MersenneTwister(SEED), length(hall))]
ish = iall[randperm(MersenneTwister(SEED), length(iall))]
test_h = hsh[1:TEST_PER_CLASS]; test_i = ish[1:TEST_PER_CLASS]
val_h  = hsh[TEST_PER_CLASS+1:TEST_PER_CLASS+VAL_PER_CLASS]
val_i  = ish[TEST_PER_CLASS+1:TEST_PER_CLASS+VAL_PER_CLASS]

println("Leads: $(collect(LEADS)) | Validation $(length(val_h))H/$(length(val_i))I | Test $(length(test_h))H/$(length(test_i))I")
println("Search: $(SEARCH_ITERATIONS) random + $(FOCUSED_TOPK)×$(FOCUSED_ITERATIONS) focused\n")

function eval_config(hp)
    H = [feature(r, hp) for r in val_h]; I = [feature(r, hp) for r in val_i]
    return cv_auc(mat(H), mat(I))
end

struct Res; auc::Float64; hp::HyperParams; end
function runsearch(params, label)
    n=length(params); res=Vector{Res}(undef,n); best=Threads.Atomic{Float64}(0.0)
    p=Progress(n; desc="$label...", showspeed=true)
    Threads.@threads for i in 1:n
        a = eval_config(params[i]); res[i]=Res(a,params[i])
        Threads.atomic_max!(best,a); next!(p; showvalues=[(:best_val_AUROC, round(best[],digits=3))])
    end
    sort!(res, by=r->r.auc, rev=true); return res
end

rng = MersenneTwister(SEED)
randp = [sample_random(rng) for _ in 1:SEARCH_ITERATIONS]
rr = runsearch(randp, "1. Random")
println("\n  Top random val AUROCs: ", join([round(r.auc,digits=3) for r in rr[1:min(5,end)]], ", "))

frng = MersenneTwister(SEED+1); fp=HyperParams[]
for i in 1:min(FOCUSED_TOPK,length(rr)); append!(fp,[sample_focused(frng,rr[i].hp) for _ in 1:FOCUSED_ITERATIONS]); end
fr = runsearch(fp, "2. Focused")
allr = sort!(vcat(rr,fr), by=r->r.auc, rev=true)
best = allr[1].hp; val_auc = allr[1].auc

# ----- held-out test -----
println("\n", "="^72, "\nHELD-OUT TEST (winner refit on validation, scored once)\n", "="^72)
Hv = mat([feature(r,best) for r in val_h]); Iv = mat([feature(r,best) for r in val_i])
m = fit_lda(Hv, Iv)
ts = Float64[]; tl = Bool[]
for r in test_h; push!(ts, score_lda(m, feature(r,best))); push!(tl, false); end
for r in test_i; push!(ts, score_lda(m, feature(r,best))); push!(tl, true);  end
pt, lo, hi = boot_auc_ci(ts, tl)
cc = confusion_counts(tl, ts .>= 0.0)
println("  Validation CV AUROC : $(round(val_auc,digits=3))")
println("  TEST AUROC          : $(round(pt,digits=3))   95% CI [$(round(lo,digits=3)), $(round(hi,digits=3))]  ← headline")
println("  TEST balanced acc   : $(round(100*balanced_acc(cc),digits=1))%   MCC $(round(mcc(cc),digits=3))")
println("  Architecture: N_in=$(best.N_in) N_hid=$(best.N_hid) N_out=$(best.N_out) | $(length(LEADS)) leads | readout=out+hid rates")
gap = val_auc - pt
gap > 0.12 && println("  ⚠ val ($(round(val_auc,digits=2))) ≫ test ($(round(pt,digits=2))): overfitting; enlarge VAL_PER_CLASS / shrink space.")

println("\n----- Best params -----")
for (nm,v) in [("Δ",best.Δ),("pulse_amp",best.pulse_amp),("ltp_rate",best.ltp_rate),("ltp_rate_out",best.ltp_rate_out),
    ("τ_s",best.τ_s),("τ_s_output",best.τ_s_output),("τ_pretrace",best.τ_pretrace),("τ_posttrace",best.τ_posttrace),
    ("R_m_input",best.R_m_input),("R_m_output",best.R_m_output),("τ_m_input",best.τ_m_input),("τ_m_output",best.τ_m_output),
    ("τ_ref_input",best.τ_ref_input),("τ_ref_output",best.τ_ref_output),("N_in",best.N_in),("N_hid",best.N_hid),
    ("N_out",best.N_out),("density",best.density),("inhib_den",best.inhib_den),("inhib_str",best.inhib_str)]
    println("const $nm = $v")
end
println("\n", "="^72, "\nDONE\n", "="^72)
