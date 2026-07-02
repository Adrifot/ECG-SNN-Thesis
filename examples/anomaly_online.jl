"""
    anomaly_online.jl

ONLINE myocardial-infarction detection with latency, scored by the SUPERVISED
LDA projection (the same discriminative direction that gives the ~0.89
classification AUROC), applied per window for low-latency detection.

Why this and not the unsupervised novelty score: the per-neuron firing-rate
features carry the MI signal in a specific, small direction of feature space. An
unsupervised "distance from baseline" score averages over all ~300 features and
dilutes that signal to nothing (it failed: anomaly lift ≈ control lift). The LDA,
trained on cross-patient labels, projects onto exactly the discriminative
direction and amplifies it. Here we reuse that projection as the online score.

Protocol:
  TRAIN (offline, once): on a *labelled* TRAINING pool, run each recording, freeze
    STDP, collect 1s windowed multi-lead feature vectors, and fit a diagonal LDA
    → discriminant direction w, centre mid.
  DEPLOY (per patient, online): build a hybrid signal (healthy → infarction at
    TRANSITION); calibrate/freeze on the healthy prefix; in 1 s windows project
    the feature vector onto w; alarm when the projection exceeds a threshold
    calibrated on the patient's own healthy-prefix projections, for K_PERSIST
    consecutive windows. Latency = alarm − TRANSITION.

The TRAINING and TEST recording pools are disjoint (no leakage).
"""

include("../modules/Layers.jl")
include("../modules/Signals.jl")
include("../modules/Registry.jl")
include("../modules/Metrics.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils, .Registry, .Metrics
using Plots, Random, Statistics, ProgressMeter
using Plots.Measures
using StatsBase: sample

gr()

# ----- optimal params (from paramsearch_routed_multilead.jl held-out test) -----
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
const density      = 0.7793537471176322
const inhib_den    = 0.2926531513089269
const inhib_str    = 0.05207908906117163
const N_in  = 20
const N_hid = 25
const N_out = 25

# ----- timing / detector config -----
const dt          = 0.001
const FS          = 1000.0
const PRE_R       = 0.25
const GAP         = 100.0
const R_IDX       = round(Int, PRE_R * FS) + 1
const LEADS       = [2, 3, 6, 8, 9, 10]
const WIN_SEC     = 1.0
const SMOOTH_W    = 3
const K_PERSIST   = 2
const Z_THRESH    = 2.5
const SLOPE_WIN   = 5          # windows for slope-based detection
const LDA_SHRINK  = 1e-3

# hybrid (deployment) timeline
const TSIM        = 60.0
const CAL_START   = 8.0
const CAL_END     = 15.0
const TRANSITION  = 16.0
# recording timeline for building the LDA training set
const TRAIN_TSIM  = 18.0
const SETTLE      = 8.0        # discard initial transient during training

const N_TRAIN_PER = 25        # labelled recordings per class for the LDA
const N_TRIALS    = 20        # test trials per condition
const SEED        = 42

# ----- detection approaches -----
const PURE_CONTROL     = true    # control = pure healthy (no stitch)
const STDP_BOOST       = 3.0     # multiply STDP rates after transition
const MAHAL_SHRINK     = 1e-3    # diagonal shrinkage for Mahalanobis

seg_base(t) = t < R_IDX - 60 ? 0 : (t <= R_IDX + 80 ? 2 : 4)

# ----- routed network -----
function build_routed(rng; lr_mult=1.0)
    inp(rev) = Neuron(rev ? "id" : "iu"; R_m=R_m_input, τ_m=τ_m_input, τ_s=τ_s, τ_ref=τ_ref_input, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace, isreverse=rev)
    hidt = Neuron("h"; R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    outt = Neuron("o"; R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    exc = Synapse(1,2; learningrate=ltp_rate*lr_mult, wmax=1.0)
    exc_out = Synapse(1,2; learningrate=ltp_rate_out*lr_mult, wmax=1.0)
    inh = Synapse(1,1; learningrate=0.0, wmax=1.0, isinhibitory=true)
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

# 6 routed pulse arrays for a PURE recording on one lead.
function single_pulses(rec, lead, nsteps)
    filt = get_filtered_signal(load_raw_signal(rec.patient, rec.session; lead=lead))
    beats = segment_beats(filt, get_R_peaks(filt; fs=FS); fs=FS)
    arr = [zeros(nsteps) for _ in 1:6]; offset = 0.0; bi = 1
    while offset < nsteps && !isempty(beats)
        beat = beats[mod1(bi, length(beats))]
        for s in delta_mod(normalize_beat(beat))
            step = Int(floor(s.time + offset)) + 1
            1 <= step <= nsteps || continue
            base = seg_base(s.time)
            s.polarity ? (arr[base+1][step] += pulse_amp) : (arr[base+2][step] -= pulse_amp)
        end
        offset += length(beat) + GAP; bi += 1
    end
    return arr
end

# 6 routed pulse arrays for a HYBRID: source prefix until transition_time, then target.
function hybrid_pulses(src_rec, tgt_rec, lead, nsteps; transition_time=TRANSITION)
    fsig = get_filtered_signal(load_raw_signal(src_rec.patient, src_rec.session; lead=lead))
    tsig = get_filtered_signal(load_raw_signal(tgt_rec.patient, tgt_rec.session; lead=lead))
    bsrc = segment_beats(fsig, get_R_peaks(fsig; fs=FS); fs=FS)
    btgt = segment_beats(tsig, get_R_peaks(tsig; fs=FS); fs=FS)
    arr = [zeros(nsteps) for _ in 1:6]; t_ms = transition_time * 1000.0
    function emit(beats, t0, tmax)
        offset = t0; bi = 1
        while offset < tmax && !isempty(beats)
            beat = beats[mod1(bi, length(beats))]
            for s in delta_mod(normalize_beat(beat))
                step = Int(floor(s.time + offset)) + 1
                1 <= step <= nsteps || continue
                base = seg_base(s.time)
                s.polarity ? (arr[base+1][step] += pulse_amp) : (arr[base+2][step] -= pulse_amp)
            end
            offset += length(beat) + GAP; bi += 1
        end
        return offset
    end
    last = emit(bsrc, 0.0, t_ms)
    emit(btgt, max(last, t_ms), nsteps * 1.0)
    return arr
end

# Per-window [output;hidden] firing-rate vectors for ONE lead.
# win_bins: number of 1-second bins to aggregate (1=non-overlapping, 3=rolling 3s window)
# freeze_at: freeze STDP after this time (Inf = never)
function lead_windows(arr, tsim; win_bins=1, freeze_at=Inf)
    nsteps = length(arr[1])
    net = build_routed(MersenneTwister(SEED))
    input_fn = t -> begin
        idx = clamp(Int(floor(t/dt)) + 1, 1, nsteps)
        v = zeros(length(net.neuronlayers)); @inbounds for k in 1:6; v[k]=arr[k][idx]; end; v
    end
    n_seconds = floor(Int, tsim)
    nwin = max(0, n_seconds - win_bins + 1)
    feats = [zeros(N_out + N_hid) for _ in 1:max(1, nwin)]
    seen_o = fill(-Inf, N_out); seen_h = fill(-Inf, N_hid)
    cnt_o = zeros(Int, N_out); cnt_h = zeros(Int, N_hid)
    buf_o = [zeros(Int, N_out) for _ in 1:win_bins]
    buf_h = [zeros(Int, N_hid) for _ in 1:win_bins]
    cur_sec = 0; cur_feat = 1
    cb = function(t, net, step)
        sec = floor(Int, t)
        if sec > cur_sec && sec <= n_seconds
            bi = mod1(sec, win_bins)
            copyto!(buf_o[bi], cnt_o); copyto!(buf_h[bi], cnt_h)
            fill!(cnt_o, 0); fill!(cnt_h, 0)
            cur_sec = sec
            if sec >= win_bins && cur_feat <= nwin
                fill!(feats[cur_feat], 0.0)
                for b in 1:win_bins
                    fi = mod1(sec - win_bins + b, win_bins)
                    for j in 1:N_out; feats[cur_feat][j] += buf_o[fi][j]; end
                    for j in 1:N_hid; feats[cur_feat][N_out + j] += buf_h[fi][j]; end
                end
                feats[cur_feat] ./= win_bins
                cur_feat += 1
            end
        end
        ol = net.neuronlayers[8]; hl = net.neuronlayers[7]
        @inbounds for j in 1:N_out
            if ol.t_lastout[j] > seen_o[j] + 1e-9; cnt_o[j]+=1; seen_o[j]=ol.t_lastout[j]; end
        end
        @inbounds for j in 1:N_hid
            if hl.t_lastout[j] > seen_h[j] + 1e-9; cnt_h[j]+=1; seen_h[j]=hl.t_lastout[j]; end
        end
    end
    runlayers!(net, dt, tsim; inputfn=input_fn, callback=cb, freeze_at=freeze_at)
    return feats[1:cur_feat-1]
end

# Extended lead_windows with weight tracking and optional boosted STDP.
function lead_windows_extended(arr, tsim; lr_mult=1.0, win_bins=1, freeze_at=Inf)
    nsteps = length(arr[1])
    net = build_routed(MersenneTwister(SEED); lr_mult=lr_mult)
    input_fn = t -> begin
        idx = clamp(Int(floor(t/dt)) + 1, 1, nsteps)
        v = zeros(length(net.neuronlayers)); @inbounds for k in 1:6; v[k]=arr[k][idx]; end; v
    end
    n_seconds = floor(Int, tsim)
    nwin = max(0, n_seconds - win_bins + 1)
    feats = [zeros(N_out + N_hid) for _ in 1:max(1, nwin)]
    wtrace = zeros(max(1, nwin))
    seen_o = fill(-Inf, N_out); seen_h = fill(-Inf, N_hid)
    cnt_o = zeros(Int, N_out); cnt_h = zeros(Int, N_hid)
    buf_o = [zeros(Int, N_out) for _ in 1:win_bins]
    buf_h = [zeros(Int, N_hid) for _ in 1:win_bins]
    cur_sec = 0; cur_feat = 1
    cb = function(t, net, step)
        sec = floor(Int, t)
        if sec > cur_sec && sec <= n_seconds
            bi = mod1(sec, win_bins)
            copyto!(buf_o[bi], cnt_o); copyto!(buf_h[bi], cnt_h)
            fill!(cnt_o, 0); fill!(cnt_h, 0)
            cur_sec = sec
            if sec >= win_bins && cur_feat <= nwin
                fill!(feats[cur_feat], 0.0); wtrace[cur_feat] = 0.0
                for b in 1:win_bins
                    fi = mod1(sec - win_bins + b, win_bins)
                    for j in 1:N_out; feats[cur_feat][j] += buf_o[fi][j]; end
                    for j in 1:N_hid; feats[cur_feat][N_out + j] += buf_h[fi][j]; end
                end
                feats[cur_feat] ./= win_bins
                wtrace[cur_feat] = mean(net.synapselayers[7].ws)
                cur_feat += 1
            end
        end
        ol = net.neuronlayers[8]; hl = net.neuronlayers[7]
        @inbounds for j in 1:N_out
            if ol.t_lastout[j] > seen_o[j] + 1e-9; cnt_o[j]+=1; seen_o[j]=ol.t_lastout[j]; end
        end
        @inbounds for j in 1:N_hid
            if hl.t_lastout[j] > seen_h[j] + 1e-9; cnt_h[j]+=1; seen_h[j]=hl.t_lastout[j]; end
        end
    end
    runlayers!(net, dt, tsim; inputfn=input_fn, callback=cb, freeze_at=freeze_at)
    return feats[1:cur_feat-1], wtrace[1:cur_feat-1]
end

# Multi-lead per-window features: arr_for_lead(lead) -> 6 pulse arrays.
# win_bins: accumulate N 1-second bins per feature (1=non-overlapping, 3=rolling 3s window)
# freeze_at: freeze STDP after this time
function multilead_windows(arr_for_lead, tsim; win_bins=1, freeze_at=Inf)
    blocks = [lead_windows(arr_for_lead(ld), tsim; win_bins=win_bins, freeze_at=freeze_at) for ld in LEADS]
    nwin = length(blocks[1])
    nwin < 1 && return Float64[], Vector{Float64}[]
    win_t = [(w - 1) + win_bins / 2 for w in 1:nwin]
    feats = [reduce(vcat, [blocks[l][w] for l in 1:length(LEADS)]) for w in 1:nwin]
    return win_t, feats
end

# Multi-lead extended: returns (win_t, feats, per_window_weight_mean_across_leads)
function multilead_windows_extended(arr_for_lead, tsim; lr_mult=1.0, win_bins=1, freeze_at=Inf)
    results = [lead_windows_extended(arr_for_lead(ld), tsim; lr_mult=lr_mult, win_bins=win_bins, freeze_at=freeze_at) for ld in LEADS]
    nwin = length(results[1][1])
    nwin < 1 && return Float64[], Vector{Float64}[], Float64[]
    win_t = [(w - 1) + win_bins / 2 for w in 1:nwin]
    blocks = [r[1] for r in results]
    wt_leads = [r[2] for r in results]
    feats = [reduce(vcat, [blocks[l][w] for l in 1:length(LEADS)]) for w in 1:nwin]
    wt_mean = [mean([wt_leads[l][w] for l in 1:length(LEADS)]) for w in 1:nwin]
    return win_t, feats, wt_mean
end

const N_FEATURES  = 30        # keep top-K most discriminative features

# ----- feature selection + diagonal LDA -----
function fisher_ratio(X0, X1)
    μ0 = vec(mean(X0, dims=1)); μ1 = vec(mean(X1, dims=1))
    v0 = vec(var(X0, dims=1));  v1 = vec(var(X1, dims=1))
    return (μ1 - μ0).^2 ./ (v0 + v1 .+ 1e-12)
end

function select_top_features(X0, X1; k=N_FEATURES)
    fr = fisher_ratio(X0, X1)
    idx = sortperm(fr; rev=true)[1:min(k, length(fr))]
    return idx, X0[:, idx], X1[:, idx]
end

function fit_lda(X0, X1; feature_idx=nothing)
    if feature_idx !== nothing
        X0s = X0[:, feature_idx]; X1s = X1[:, feature_idx]
    else
        X0s = X0; X1s = X1
    end
    μ0 = vec(mean(X0s, dims=1)); μ1 = vec(mean(X1s, dims=1))
    n0, n1 = size(X0s,1), size(X1s,1)
    vp = ((n0-1).*vec(var(X0s,dims=1)) .+ (n1-1).*vec(var(X1s,dims=1))) ./ max(1,n0+n1-2) .+ LDA_SHRINK
    return (w=(μ1.-μ0)./vp, mid=(μ0.+μ1)./2)    # project>0 ⇒ infarction side
end

project(m, x; feature_idx=nothing) = feature_idx === nothing ?
    sum(m.w .* (x .- m.mid)) : sum(m.w .* (x[feature_idx] .- m.mid))

matrows(V) = (d=length(V[1]); reduce(vcat, [reshape(v,1,d) for v in V]))

# Collect labelled post-settle windowed features from pure recordings (STDP always active).
function training_rows(recs)
    rows = Vector{Float64}[]
    @showprogress for r in recs
        wt, feats = multilead_windows(ld -> single_pulses(r, ld, round(Int, TRAIN_TSIM/dt)), TRAIN_TSIM)
        for i in eachindex(wt)
            wt[i] > SETTLE && push!(rows, feats[i])
        end
    end
    return rows
end

# Collect features from HYBRID recordings with ACTIVE STDP throughout.
# Each hybrid: source_healthy → target, transition at TRAIN_SWITCH.
# Post-settle features capture the adaptation trajectory TOWARD the target,
# starting from a healthy-adapted network state — matching deployment exactly.
function training_rows_hybrid(target_recs, source_recs; switch_time=8.0)
    rows = Vector{Float64}[]
    nsrc = length(source_recs)
    @showprogress for (i, tgt) in enumerate(target_recs)
        src = source_recs[mod1(i, nsrc)]
        wt, feats = multilead_windows(
            ld -> hybrid_pulses(src, tgt, ld, round(Int, TRAIN_TSIM/dt); transition_time=switch_time),
            TRAIN_TSIM)
        for j in eachindex(wt)
            wt[j] > SETTLE && push!(rows, feats[j])
        end
    end
    return rows
end

smooth(v, w) = [mean(v[max(1,i-w÷2):min(end,i+w÷2)]) for i in eachindex(v)]

# Online detection: project each window onto the LDA direction; threshold from the
# patient's own healthy-prefix projections. Returns (proj, thr, alarm_time, zsep).
# Also returns slope-based detection results.
function detect(win_t, feats, lda, feat_idx)
    proj = smooth([project(lda, f; feature_idx=feat_idx) for f in feats], SMOOTH_W)
    cal = [i for i in eachindex(win_t) if CAL_START <= win_t[i] <= CAL_END]
    μc = mean(proj[cal]); σc = std(proj[cal]) + 1e-9
    thr = μc + Z_THRESH * σc

    # threshold-based detection
    alarm_val = nothing; run = 0
    for i in eachindex(proj)
        win_t[i] <= CAL_END && continue
        run = proj[i] > thr ? run + 1 : 0
        if run >= K_PERSIST; alarm_val = win_t[i - K_PERSIST + 1]; break; end
    end
    post = [proj[i] for i in eachindex(win_t) if win_t[i] > TRANSITION]
    zsep = isempty(post) ? NaN : (mean(post) - μc) / σc

    # slope-based detection: alarm when projection consistently drifts upward
    # slope over SLOPE_WIN windows
    slopes = [i > SLOPE_WIN ? (proj[i] - proj[i-SLOPE_WIN]) / SLOPE_WIN : NaN for i in eachindex(proj)]
    cal_slopes = [slopes[i] for i in cal if !isnan(slopes[i])]
    μs = mean(cal_slopes); σs = std(cal_slopes) + 1e-9
    thr_slope = μs + Z_THRESH * σs
    alarm_slope = nothing; run_s = 0
    for i in eachindex(slopes)
        (win_t[i] <= CAL_END || isnan(slopes[i])) && continue
        run_s = slopes[i] > thr_slope ? run_s + 1 : 0
        if run_s >= K_PERSIST; alarm_slope = win_t[i - K_PERSIST + 1]; break; end
    end

    return (proj=proj, thr=thr, alarm_val=alarm_val, zsep=zsep,
            alarm_slope=alarm_slope, slopes=slopes, thr_slope=thr_slope)
end

function detect_mahalanobis(win_t, feats)
    cal = [i for i in eachindex(win_t) if CAL_START <= win_t[i] <= CAL_END]
    cal_mat = matrows(feats[cal])
    μc = vec(mean(cal_mat, dims=1))
    σc = vec(std(cal_mat, dims=1)) .+ sqrt(MAHAL_SHRINK)
    scores = [sqrt(sum(((f .- μc) ./ σc).^2)) for f in feats]
    cal_scores = [scores[i] for i in cal]
    thr = mean(cal_scores) + Z_THRESH * std(cal_scores)
    alarm_val = nothing; run = 0
    for i in eachindex(scores)
        win_t[i] <= CAL_END && continue
        run = scores[i] > thr ? run + 1 : 0
        if run >= K_PERSIST; alarm_val = win_t[i - K_PERSIST + 1]; break; end
    end
    post = [scores[i] for i in eachindex(win_t) if win_t[i] > TRANSITION]
    zsep = isempty(post) ? NaN : (mean(post) - mean(cal_scores)) / (std(cal_scores) + 1e-9)
    return (scores=scores, thr=thr, alarm_val=alarm_val, zsep=zsep, norm_post=post)
end

function detect_weight(win_t, wtrace)
    cal = [i for i in eachindex(win_t) if CAL_START <= win_t[i] <= CAL_END]
    μc = mean(wtrace[cal]); σc = std(wtrace[cal]) + 1e-9
    thr = μc - Z_THRESH * σc
    alarm_val = nothing; run = 0
    for i in eachindex(wtrace)
        win_t[i] <= CAL_END && continue
        run = wtrace[i] < thr ? run + 1 : 0
        if run >= K_PERSIST; alarm_val = win_t[i - K_PERSIST + 1]; break; end
    end
    post = [wtrace[i] for i in eachindex(win_t) if win_t[i] > TRANSITION]
    zsep = isempty(post) ? NaN : (μc - mean(post)) / σc
    return (wtrace=wtrace, thr=thr, alarm_val=alarm_val, zsep=zsep)
end

pure_healthy_windows(rec, tsim; kw...) = multilead_windows(ld -> single_pulses(rec, ld, round(Int, tsim/dt)), tsim; kw...)

# -------------------------------------------------------------------------------------------
Random.seed!(SEED)
labelled   = filter(r -> r.label != :unknown, build_registry("./ecg-db"))
healthy    = filter(r -> r.label == :healthy,    labelled)
infarction = filter(r -> r.label == :infarction, labelled)
hsh = healthy[randperm(MersenneTwister(SEED), length(healthy))]
ish = infarction[randperm(MersenneTwister(SEED+1), length(infarction))]
# disjoint train / test pools
h_train = hsh[1:N_TRAIN_PER];                 h_test = hsh[N_TRAIN_PER+1:end]
i_train = ish[1:N_TRAIN_PER];                 i_test = ish[N_TRAIN_PER+1:end]

println("Training LDA on $(N_TRAIN_PER) healthy + $(N_TRAIN_PER) infarction recordings (pure STDP, post-settle windows)…")
Xh_full = matrows(training_rows(h_train))
Xi_full = matrows(training_rows(i_train))
feat_idx, Xh, Xi = select_top_features(Xh_full, Xi_full)
lda = fit_lda(Xh, Xi)
println("  trained: $(size(Xh_full,2)) raw dim → $(length(feat_idx)) selected features, $(size(Xh,1)) healthy + $(size(Xi,1)) infarction windows")

# ----- single-trial DEMO -----
Random.seed!(SEED + 7)
h0 = sample(h_test); a0 = sample(i_test)
println("\nDEMO  healthy=$(h0.patient)  anomalous=$(a0.patient)")
wt, ft = multilead_windows(ld -> hybrid_pulses(h0, a0, ld, round(Int, TSIM/dt)), TSIM)
d = detect(wt, ft, lda, feat_idx)
lat = d.alarm_val === nothing ? NaN : d.alarm_val - TRANSITION
lat_s = d.alarm_slope === nothing ? NaN : d.alarm_slope - TRANSITION
println("  thr-based:  alarm=$(d.alarm_val===nothing ? "none" : "$(round(d.alarm_val,digits=1))s")  latency=$(isnan(lat) ? "—" : "$(round(lat,digits=1))s")")
println("  slope-based: alarm=$(d.alarm_slope===nothing ? "none" : "$(round(d.alarm_slope,digits=1))s")  latency=$(isnan(lat_s) ? "—" : "$(round(lat_s,digits=1))s")")
println("  per-window separation: post-transition projection is $(round(d.zsep,digits=2))σ above the healthy band")
println("  post-transition slope: $(round(mean(skipmissing(d.slopes[wt .> TRANSITION])),digits=4)) σ/s (cal slope $(round(mean(skipmissing(d.slopes[(CAL_START .<= wt .<= CAL_END)])),digits=4)))")

p = plot(wt, d.proj; lw=2, color=:navy, label="LDA projection", xlabel="time (s)",
         ylabel="discriminant score", title="Online MI detection (LDA projection, single trial)",
         titlefontsize=11, size=(880,440), margin=5mm, legend=:topleft)
vspan!(p, [CAL_START, CAL_END]; alpha=0.10, color=:green, label="calibration")
hline!(p, [d.thr]; ls=:dash, color=:gray, label="thr threshold")
hline!(p, [d.thr_slope]; ls=:dashdot, color=:orange, label="slope threshold")
vline!(p, [TRANSITION]; lw=2, color=:black, label="true transition")
d.alarm_val !== nothing && vline!(p, [d.alarm_val]; lw=2, color=:orange, label="alarm (thr)")
d.alarm_slope !== nothing && vline!(p, [d.alarm_slope]; lw=2, color=:red, label="alarm (slope)")
demo_out = joinpath(@__DIR__, "../docs/imgs/anomaly_online_demo.png")
isdir(dirname(demo_out)) || mkpath(dirname(demo_out))
savefig(p, demo_out); println("  demo plot -> $(demo_out)")

# ----- multi-trial STATISTICS -----
function run_trials_lda(n, gen)
    thr_detected = Bool[]; thr_lats = Float64[]; thr_prem = 0
    slp_detected = Bool[]; slp_lats = Float64[]; slp_prem = 0
    zseps = Float64[]; cont_scores = Float64[]; peak_scores = Float64[]
    @showprogress "  LDA trials…" for _ in 1:n
        h, a = gen()
        wt, ft = multilead_windows(ld -> hybrid_pulses(h, a, ld, round(Int, TSIM/dt)), TSIM)
        d = detect(wt, ft, lda, feat_idx)
        d.zsep !== nothing && !isnan(d.zsep) && push!(zseps, d.zsep)
        post_z = [d.proj[i] for i in eachindex(wt) if wt[i] > TRANSITION]
        cal_z = [d.proj[i] for i in eachindex(wt) if CAL_START <= wt[i] <= CAL_END]
        μc = mean(cal_z); σc = std(cal_z) + 1e-9
        push!(cont_scores, (mean(post_z) - μc) / σc)
        push!(peak_scores, (maximum(post_z) - μc) / σc)
        if d.alarm_val === nothing
            push!(thr_detected, false)
        elseif d.alarm_val >= TRANSITION - WIN_SEC
            push!(thr_detected, true); push!(thr_lats, d.alarm_val - TRANSITION)
        else
            push!(thr_detected, false); thr_prem += 1
        end
        if d.alarm_slope === nothing
            push!(slp_detected, false)
        elseif d.alarm_slope >= TRANSITION - WIN_SEC
            push!(slp_detected, true); push!(slp_lats, d.alarm_slope - TRANSITION)
        else
            push!(slp_detected, false); slp_prem += 1
        end
    end
    return (thr_detected=thr_detected, thr_lats=thr_lats, thr_prem=thr_prem,
            slp_detected=slp_detected, slp_lats=slp_lats, slp_prem=slp_prem,
            zseps=zseps, cont_scores=cont_scores, peak_scores=peak_scores)
end

function run_trials_mahalanobis(n, gen)
    scores_list = Float64[]; alarm_list = Bool[]; lat_list = Float64[]; prem = 0
    @showprogress "  Mahalanobis trials…" for _ in 1:n
        h, a = gen()
        wt, ft = multilead_windows(ld -> hybrid_pulses(h, a, ld, round(Int, TSIM/dt)), TSIM)
        d = detect_mahalanobis(wt, ft)
        cal_scores = [d.scores[i] for i in eachindex(wt) if CAL_START <= wt[i] <= CAL_END]
        post_scores = [d.scores[i] for i in eachindex(wt) if wt[i] > TRANSITION]
        push!(scores_list, isempty(post_scores) ? NaN : (mean(post_scores) - mean(cal_scores)) / (std(cal_scores) + 1e-9))
        if d.alarm_val === nothing
            push!(alarm_list, false)
        elseif d.alarm_val >= TRANSITION - WIN_SEC
            push!(alarm_list, true); push!(lat_list, d.alarm_val - TRANSITION)
        else
            push!(alarm_list, false); prem += 1
        end
    end
    return (scores=scores_list, alarms=alarm_list, lats=lat_list, prem=prem)
end

function run_trials_boosted_lda(n, gen)
    thr_detected = Bool[]; thr_lats = Float64[]; thr_prem = 0
    zseps = Float64[]; cont_scores = Float64[]
    @showprogress "  Boosted STDP trials…" for _ in 1:n
        h, a = gen()
        wt, ft, _ = multilead_windows_extended(ld -> hybrid_pulses(h, a, ld, round(Int, TSIM/dt)), TSIM; lr_mult=STDP_BOOST)
        d = detect(wt, ft, lda, feat_idx)
        d.zsep !== nothing && !isnan(d.zsep) && push!(zseps, d.zsep)
        post_z = [d.proj[i] for i in eachindex(wt) if wt[i] > TRANSITION]
        cal_z = [d.proj[i] for i in eachindex(wt) if CAL_START <= wt[i] <= CAL_END]
        μc = mean(cal_z); σc = std(cal_z) + 1e-9
        push!(cont_scores, (mean(post_z) - μc) / σc)
        if d.alarm_val === nothing
            push!(thr_detected, false)
        elseif d.alarm_val >= TRANSITION - WIN_SEC
            push!(thr_detected, true); push!(thr_lats, d.alarm_val - TRANSITION)
        else
            push!(thr_detected, false); thr_prem += 1
        end
    end
    return (thr_detected=thr_detected, thr_lats=thr_lats, thr_prem=thr_prem, zseps=zseps, cont_scores=cont_scores)
end

function run_trials_frozen_lda(n, gen)
    thr_detected = Bool[]; thr_lats = Float64[]; thr_prem = 0
    zseps = Float64[]; cont_scores = Float64[]
    @showprogress "  Frozen LDA+overlap trials…" for _ in 1:n
        h, a = gen()
        wt, ft = multilead_windows(ld -> hybrid_pulses(h, a, ld, round(Int, TSIM/dt)), TSIM;
                                   win_bins=3, freeze_at=CAL_END)
        d = detect(wt, ft, lda, feat_idx)
        d.zsep !== nothing && !isnan(d.zsep) && push!(zseps, d.zsep)
        post_z = [d.proj[i] for i in eachindex(wt) if wt[i] > TRANSITION]
        cal_z = [d.proj[i] for i in eachindex(wt) if CAL_START <= wt[i] <= CAL_END]
        μc = mean(cal_z); σc = std(cal_z) + 1e-9
        push!(cont_scores, (mean(post_z) - μc) / σc)
        if d.alarm_val === nothing
            push!(thr_detected, false)
        elseif d.alarm_val >= TRANSITION
            push!(thr_detected, true); push!(thr_lats, d.alarm_val - TRANSITION)
        else
            push!(thr_detected, false); thr_prem += 1
        end
    end
    return (thr_detected=thr_detected, thr_lats=thr_lats, thr_prem=thr_prem, zseps=zseps, cont_scores=cont_scores)
end

function run_trials_weight(n, gen)
    alarms = Bool[]; lats = Float64[]; prem = 0; zseps = Float64[]
    @showprogress "  Weight trials…" for _ in 1:n
        h, a = gen()
        wt, _, wtrace = multilead_windows_extended(ld -> hybrid_pulses(h, a, ld, round(Int, TSIM/dt)), TSIM)
        d = detect_weight(wt, wtrace)
        d.zsep !== nothing && !isnan(d.zsep) && push!(zseps, d.zsep)
        if d.alarm_val === nothing
            push!(alarms, false)
        elseif d.alarm_val >= TRANSITION - WIN_SEC
            push!(alarms, true); push!(lats, d.alarm_val - TRANSITION)
        else
            push!(alarms, false); prem += 1
        end
    end
    return (alarms=alarms, lats=lats, prem=prem, zseps=zseps)
end

function distinct_healthy_pair()
    a = sample(h_test); b = sample(h_test)
    while b.patient == a.patient; b = sample(h_test); end
    return (a, b)
end

function run_controls_hybrid(n)
    thr_alarms = Bool[]; slp_alarms = Bool[]
    cont_scores = Float64[]; peak_scores = Float64[]
    @showprogress "  Control (hybrid)…" for _ in 1:n
        h, b = distinct_healthy_pair()
        wt, ft = multilead_windows(ld -> hybrid_pulses(h, b, ld, round(Int, TSIM/dt)), TSIM)
        d = detect(wt, ft, lda, feat_idx)
        push!(thr_alarms, d.alarm_val !== nothing)
        push!(slp_alarms, d.alarm_slope !== nothing)
        post_z = [d.proj[i] for i in eachindex(wt) if wt[i] > TRANSITION]
        cal_z = [d.proj[i] for i in eachindex(wt) if CAL_START <= wt[i] <= CAL_END]
        μc = mean(cal_z); σc = std(cal_z) + 1e-9
        push!(cont_scores, (mean(post_z) - μc) / σc)
        push!(peak_scores, (maximum(post_z) - μc) / σc)
    end
    return (thr_alarms=thr_alarms, slp_alarms=slp_alarms, cont_scores=cont_scores, peak_scores=peak_scores)
end

function run_controls_pure(n)
    thr_alarms = Bool[]; mahal_alarms = Bool[]
    cont_scores = Float64[]; mahal_scores = Float64[]
    @showprogress "  Control (pure)…" for _ in 1:n
        h = sample(h_test)
        wt, ft = pure_healthy_windows(h, TSIM)
        d_lda = detect(wt, ft, lda, feat_idx)
        d_mah = detect_mahalanobis(wt, ft)
        push!(thr_alarms, d_lda.alarm_val !== nothing)
        push!(mahal_alarms, d_mah.alarm_val !== nothing)
        post_z = [d_lda.proj[i] for i in eachindex(wt) if wt[i] > TRANSITION]
        cal_z = [d_lda.proj[i] for i in eachindex(wt) if CAL_START <= wt[i] <= CAL_END]
        μc = mean(cal_z); σc = std(cal_z) + 1e-9
        push!(cont_scores, isempty(post_z) ? NaN : (mean(post_z) - μc) / σc)
        mahal_post = [d_mah.scores[i] for i in eachindex(wt) if wt[i] > TRANSITION]
        mahal_cal = [d_mah.scores[i] for i in eachindex(wt) if CAL_START <= wt[i] <= CAL_END]
        push!(mahal_scores, isempty(mahal_post) ? NaN : (mean(mahal_post) - mean(mahal_cal)) / (std(mahal_cal) + 1e-9))
    end
    return (thr_alarms=thr_alarms, mahal_alarms=mahal_alarms, cont_scores=cont_scores, mahal_scores=mahal_scores)
end

function run_controls_boosted(n)
    alarmed = Bool[]
    @showprogress "  Control (boosted)…" for _ in 1:n
        h = sample(h_test)
        wt, ft, _ = multilead_windows_extended(ld -> single_pulses(h, ld, round(Int, TSIM/dt)), TSIM; lr_mult=STDP_BOOST)
        d = detect(wt, ft, lda, feat_idx)
        push!(alarmed, d.alarm_val !== nothing)
    end
    return alarmed
end

function run_controls_weight(n)
    alarmed = Bool[]
    @showprogress "  Control (weight)…" for _ in 1:n
        h = sample(h_test)
        wt, _, wtrace = multilead_windows_extended(ld -> single_pulses(h, ld, round(Int, TSIM/dt)), TSIM)
        d = detect_weight(wt, wtrace)
        push!(alarmed, d.alarm_val !== nothing)
    end
    return alarmed
end

function run_controls_frozen(n)
    thr_alarms = Bool[]
    @showprogress "  Control (frozen)…" for _ in 1:n
        h = sample(h_test)
        wt, ft = pure_healthy_windows(h, TSIM; win_bins=3, freeze_at=CAL_END)
        d = detect(wt, ft, lda, feat_idx)
        push!(thr_alarms, d.alarm_val !== nothing)
    end
    return thr_alarms
end

med(x) = isempty(x) ? NaN : sort(x)[max(1, ceil(Int, 0.5*length(x)))]
qt(x,p) = isempty(x) ? NaN : sort(x)[clamp(ceil(Int, p*length(x)), 1, length(x))]

println("\nRunning $(N_TRIALS) anomaly trials + $(N_TRIALS) control trials per method…")

# ----- 1. LDA projection (baseline) -----
println("\n--- LDA projection (baseline STDP) ---")
pos_lda = run_trials_lda(N_TRIALS, () -> (sample(h_test), sample(i_test)))
ctrl_lda = PURE_CONTROL ? run_controls_pure(N_TRIALS) : run_controls_hybrid(N_TRIALS)

# ----- 2. LDA + boosted STDP -----
println("\n--- LDA + boosted STDP ($(STDP_BOOST)× post-transition) ---")
pos_boost = run_trials_boosted_lda(N_TRIALS, () -> (sample(h_test), sample(i_test)))
ctrl_boost = run_controls_boosted(N_TRIALS)

# ----- 3. Mahalanobis distance -----
println("\n--- Mahalanobis distance (raw features, no LDA) ---")
pos_mahal = run_trials_mahalanobis(N_TRIALS, () -> (sample(h_test), sample(i_test)))

# ----- 4. Weight monitoring -----
println("\n--- Weight monitoring (hidden→output mean) ---")
pos_wt = run_trials_weight(N_TRIALS, () -> (sample(h_test), sample(i_test)))
ctrl_wt = run_controls_weight(N_TRIALS)

# ----- 5. LDA + frozen STDP + overlapping windows -----
println("\n--- LDA + frozen STDP (3s overlapping windows) ---")
pos_frozen = run_trials_frozen_lda(N_TRIALS, () -> (sample(h_test), sample(i_test)))
ctrl_frozen = run_controls_frozen(N_TRIALS)

# ----- RESULTS TABLE -----
println("\n", "="^75)
println("ONLINE DETECTION COMPARISON  (TSIM=$(TSIM)s, Z_THRESH=$(Z_THRESH), K=$(K_PERSIST))")
if PURE_CONTROL
    println("Control = pure healthy (no stitch)")
else
    println("Control = healthy→healthy hybrid")
end
println("="^75)
println(rpad("Approach", 22), rpad("Det%", 8), rpad("Latency", 14), rpad("FAR%", 8), rpad("zsep", 10))
println("-"^75)

function summary_line(label, det, lats, far, zsep)
    dr = round(100*mean(det), digits=1)
    md = isempty(lats) ? "—" : "$(round(med(lats),digits=1))s"
    far_str = round(100*mean(far), digits=1)
    zs = round(mean(zsep), digits=2)
    println(rpad(label, 22), rpad("$(dr)%", 8), rpad("$(md)", 14), rpad("$(far_str)%", 8), rpad("$(zs)σ", 10))
end

# LDA
ctrl_alarms = PURE_CONTROL ? ctrl_lda.thr_alarms : ctrl_lda.thr_alarms
summary_line("LDA thr", pos_lda.thr_detected, pos_lda.thr_lats, ctrl_alarms, pos_lda.zseps)
summary_line("LDA slope", pos_lda.slp_detected, pos_lda.slp_lats, PURE_CONTROL ? ctrl_lda.thr_alarms : ctrl_lda.slp_alarms, pos_lda.zseps)

# Boosted
ctrl_boost_far = PURE_CONTROL ? ctrl_boost : ctrl_boost
summary_line("LDA + boosted STDP", pos_boost.thr_detected, pos_boost.thr_lats, ctrl_boost_far, pos_boost.zseps)

# Mahalanobis
ctrl_mahal_far = PURE_CONTROL ? ctrl_lda.mahal_alarms : ctrl_lda.thr_alarms
mahal_zsep = Float64[]
for _ in 1:N_TRIALS
    h, a = sample(h_test), sample(i_test)
    wt, ft = multilead_windows(ld -> hybrid_pulses(h, a, ld, round(Int, TSIM/dt)), TSIM)
    d = detect_mahalanobis(wt, ft)
    !isnan(d.zsep) && push!(mahal_zsep, d.zsep)
end
summary_line("Mahalanobis", pos_mahal.alarms, pos_mahal.lats, ctrl_mahal_far, mahal_zsep)

# Frozen STDP + overlap
summary_line("LDA frozen+overlap", pos_frozen.thr_detected, pos_frozen.thr_lats, ctrl_frozen, pos_frozen.zseps)

# Weight
summary_line("Weight drop", pos_wt.alarms, pos_wt.lats, ctrl_wt, pos_wt.zseps)

println("-"^75)

# LDA frozen continuous separation
println("  Frozen+overlap μ(mean z) = $(round(mean(pos_frozen.cont_scores),digits=2))σ  FAR = $(round(100*mean(ctrl_frozen),digits=1))%")

# ----- AUROC for LDA (continuous score) -----
auc_ctrl_scores = PURE_CONTROL ? ctrl_lda.cont_scores : ctrl_lda.cont_scores
roc_auc_mean = auroc(vcat(pos_lda.cont_scores, auc_ctrl_scores),
                     vcat(fill(true, N_TRIALS), fill(false, N_TRIALS)))
println("\n  LDA continuous-score AUROC(mean) = $(round(roc_auc_mean,digits=3))")
if PURE_CONTROL
    println("    Anomaly μ(mean z) = $(round(mean(pos_lda.cont_scores),digits=2))σ  Control μ = $(round(mean(auc_ctrl_scores),digits=2))σ")
end

# Mahalanobis continuous-score AUROC
ctrl_mahal_scores = PURE_CONTROL ? ctrl_lda.mahal_scores : Float64[]
if !isempty(ctrl_mahal_scores) && length(ctrl_mahal_scores)==length(pos_mahal.scores)
    mahal_auc = auroc(vcat(pos_mahal.scores, ctrl_mahal_scores),
                      vcat(fill(true, N_TRIALS), fill(false, N_TRIALS)))
    println("  Mahalanobis AUROC(mean) = $(round(mahal_auc,digits=3))")
    println("    Anomaly μ(mean z) = $(round(mean(pos_mahal.scores),digits=2))σ  Control μ = $(round(mean(ctrl_mahal_scores),digits=2))σ")
end

println("\n  Leads: $(LEADS)  |  train/test pools disjoint  |  $(N_TRAIN_PER) recordings/class for LDA")
println("="^75)
