include("../modules/Layers.jl")
include("../modules/Signals.jl")
include("../modules/Registry.jl")
include("../modules/Metrics.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils,
      .Registry, .Metrics
using Random, Statistics, ProgressMeter, DelimitedFiles

# ----- Hyper Params -----

struct HyperParams
    Δ::Float64; pulse_amp::Float64; ltp_rate::Float64; ltp_rate_out::Float64
    τ_s::Float64; τ_s_output::Float64; τ_pretrace::Float64; τ_posttrace::Float64
    R_m_input::Float64; R_m_output::Float64; τ_m_input::Float64; τ_m_output::Float64
    τ_ref_input::Float64; τ_ref_output::Float64
    N_in::Int; N_hid::Int; N_out::Int
    density::Float64; inhib_den::Float64; inhib_str::Float64
end

const HP = HyperParams(
    0.12970001789262522, 170.48121762522024, 0.18935145456196095, 0.08378635445090134,
    42.831648889100016, 25.003717150125617, 34.510086127922726, 47.00571781147488,
    1.0674163386846072, 2.8737140296290087, 9.41794217755773, 2.4038829712094394,
    7.209849285084017, 0.5566821677223849,
    20, 25, 25,
    0.7793537471176322, 0.2926531513089269, 0.05207908906117163,
)

param_hash(hp, salt=0) = mod(hash((hp.Δ,hp.pulse_amp,hp.ltp_rate,hp.ltp_rate_out,hp.τ_s,hp.τ_s_output,
    hp.τ_pretrace,hp.τ_posttrace,hp.R_m_input,hp.R_m_output,hp.τ_m_input,hp.τ_m_output,hp.τ_ref_input,
    hp.τ_ref_output,hp.N_in,hp.N_hid,hp.N_out,hp.density,hp.inhib_den,hp.inhib_str)) + salt, typemax(Int32))

# ----- Config -----
const dt = 0.001
const READOUT_TIME = 12.0
const SETTLE = 5.0
const SAMPLE_INT = 0.2
const FS = 1000.0
const PRE_R = 0.25
const GAP = 100.0
const R_IDX = round(Int, PRE_R * FS) + 1
const SEED = 67          
const VAL_PER_CLASS = 15
const TEST_PER_CLASS = 60
const LDA_SHRINK = 1e-3
# 6 diagnostically-spread leads: II, III, aVF (inferior) + V2, V3, V4 (anterior).
const LEADS = [2, 3, 6, 8, 9, 10]

const CACHE_DIR = joinpath(@__DIR__, "cache")

seg_base(t) = t < R_IDX - 60 ? 0 : (t <= R_IDX + 80 ? 2 : 4)   # P=0, QRS=2, T=4

# ----- Network + Encoding -----
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
function feature(rec, hp=HP)
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

# ----- LDA -----
function fit_lda(X0, X1)
    μ0=vec(mean(X0,dims=1)); μ1=vec(mean(X1,dims=1)); n0,n1=size(X0,1),size(X1,1)
    vp=((n0-1).*vec(var(X0,dims=1)).+(n1-1).*vec(var(X1,dims=1)))./max(1,n0+n1-2).+LDA_SHRINK
    return (w=(μ1.-μ0)./vp, mid=(μ0.+μ1)./2)
end
score_lda(m,x)=sum(m.w.*(x.-m.mid))
mat(V) = (d=length(V[1]); reduce(vcat,[reshape(v,1,d) for v in V]))

# ----- Cached Feature EXtraction -----
# CSV layout: first column = label (0 healthy / 1 infarction), remaining = feature vector.
function _save_features(path, X, labels)
    isdir(CACHE_DIR) || mkpath(CACHE_DIR)
    writedlm(path, hcat(Float64.(labels), X), ',')
end
function _load_features(path)
    raw = readdlm(path, ',', Float64)
    labels = Bool.(raw[:, 1] .> 0.5)
    X = raw[:, 2:end]
    return X, labels
end

"""
    build_feature_cache(; force=false)

Returns `(Xval, yval, Xtest, ytest)` feature matrices, computing and caching them on
first use. `yval/ytest` are `Bool` (true = infarction).
"""
function build_feature_cache(; force::Bool=false)
    valpath  = joinpath(CACHE_DIR, "val_features.csv")
    testpath = joinpath(CACHE_DIR, "test_features.csv")
    if !force && isfile(valpath) && isfile(testpath)
        Xv, yv = _load_features(valpath); Xt, yt = _load_features(testpath)
        return Xv, yv, Xt, yt
    end

    Random.seed!(SEED)
    labelled = filter(r -> r.label != :unknown, build_registry("./ecg-db"))
    hall = filter(r -> r.label == :healthy, labelled)
    iall = filter(r -> r.label == :infarction, labelled)
    hsh = hall[randperm(MersenneTwister(SEED), length(hall))]
    ish = iall[randperm(MersenneTwister(SEED), length(iall))]
    test_h = hsh[1:TEST_PER_CLASS]; test_i = ish[1:TEST_PER_CLASS]
    val_h  = hsh[TEST_PER_CLASS+1:TEST_PER_CLASS+VAL_PER_CLASS]
    val_i  = ish[TEST_PER_CLASS+1:TEST_PER_CLASS+VAL_PER_CLASS]

    println("Extracting validation features ($(length(val_h))H + $(length(val_i))I)…")
    Hv = @showprogress [feature(r) for r in val_h]
    Iv = @showprogress [feature(r) for r in val_i]
    println("Extracting held-out test features ($(length(test_h))H + $(length(test_i))I)…")
    Ht = @showprogress [feature(r) for r in test_h]
    It = @showprogress [feature(r) for r in test_i]

    Xval  = vcat(mat(Hv), mat(Iv)); yval  = vcat(falses(length(Hv)), trues(length(Iv)))
    Xtest = vcat(mat(Ht), mat(It)); ytest = vcat(falses(length(Ht)), trues(length(It)))
    _save_features(valpath, Xval, yval); _save_features(testpath, Xtest, ytest)
    println("Cached features to $(CACHE_DIR)")
    return Xval, yval, Xtest, ytest
end

"""
    heldout_scores(; force=false)

Refit the diagonal LDA on the validation features and score the held-out test set.
Returns a named tuple `(scores, labels, model, Xval, yval, Xtest, ytest, auc)`.
`scores` and `labels` are aligned; `labels` true = infarction.
"""
function heldout_scores(; force::Bool=false)
    Xval, yval, Xtest, ytest = build_feature_cache(; force=force)
    m = fit_lda(Xval[.!yval, :], Xval[yval, :])
    scores = [score_lda(m, Xtest[i, :]) for i in 1:size(Xtest,1)]
    a = auroc(scores, ytest)
    return (scores=scores, labels=ytest, model=m,
            Xval=Xval, yval=yval, Xtest=Xtest, ytest=ytest, auc=a)
end

if abspath(PROGRAM_FILE) == @__FILE__
    r = heldout_scores()
    cc = confusion_counts(r.labels, r.scores .>= 0.0)
    println("Held-out TEST AUROC = $(round(r.auc, digits=3)) | balanced acc = ",
            "$(round(100*balanced_acc(cc), digits=1))% | MCC = $(round(mcc(cc), digits=3))")
end
