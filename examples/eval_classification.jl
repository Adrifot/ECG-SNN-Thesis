"""
    eval_classification.jl

Trial-level, imbalance-aware evaluation of the ECG anomaly-detection network.

Each hybrid recording is ONE labelled trial:
    - POSITIVE (anomaly):  healthy → infarction hybrid => the network SHOULD alarm
    - NEGATIVE (control):  healthy → healthy hybrid => the network should NOT alarm

The per-trial "alarm fired?" Boolean becomes the prediction, which permits building a
proper 2×2 confusion matrix and compute balanced accuracy, Matthews correlation
coefficient (MCC), per-class precision/recall, geometric mean, Youden's J,
Cohen's κ, F1/F2, AUROC and average precision (AUPRC) — with bootstrap and Wilson
confidence intervals.
"""

include("../modules/Layers.jl")
include("../modules/Signals.jl")
include("../modules/Registry.jl")
include("../modules/Metrics.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils,
      .Registry, .Metrics
using Statistics, Random, ProgressMeter, Plots
using StatsBase: sample

gr()   

# ----- Best params -----
const Δ            = 0.07
const ltp_rate     = 0.11
const ltp_rate_out = 0.13
const τ_s          = 27.71
const τ_m_input    = 5.96
const τ_ref_output = 1.32
const inhib_str    = 0.32
const pulse_amp    = 162.19
const τ_pretrace   = 9.47
const R_m_input    = 5.32
const R_m_output   = 1.44
const τ_m_output   = 8.95
const τ_ref_input  = 2.5
const inhib_den    = 0.6
const τ_s_output   = 18.27
const N            = 25
const density      = 0.89
const τ_posttrace  = 57.41

const dt   = 0.001
const tsim = 25.0
const SEED = 42

# ----- Evaluation configuration -----
const N_POS      = 30      # healthy→infarction trials (positive / anomaly)
const N_NEG      = 30      # healthy→healthy   trials (negative / control)
const N_BOOT     = 2000    # bootstrap resamples for CIs
const ALPHA      = 0.05    # 95% confidence intervals
const SETTLE     = 5.0     # seconds of convergence ignored before calibration
const SAMPLE_INT = 0.2     # readout sampling interval (s)

# ------------------------------------------------------------------
# Network (copied from anomaly_detection.jl
# -------------------------------------------------------------------
function build_network()
    Random.seed!(SEED)

    up_template   = Neuron("up_input";
                    R_m=R_m_input, τ_m=τ_m_input, τ_s=τ_s, τ_ref=τ_ref_input,
                    τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    down_template = Neuron("down_input";
                    R_m=R_m_input, τ_m=τ_m_input, τ_s=τ_s, τ_ref=τ_ref_input,
                    τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace, isreverse=true)
    hidden_template = Neuron("hidden";
                    R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output,
                    τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    out_template  = Neuron("output";
                    R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output,
                    τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)

    exc_syn      = Synapse(1, 2; learningrate=ltp_rate,     wmax=1.0)
    exc_syn_out  = Synapse(1, 2; learningrate=ltp_rate_out, wmax=1.0)
    inhib_tmpl   = Synapse(1, 1; learningrate=0.0, wmax=1.0, isinhibitory=true)

    up_layer     = NeuronLayer(N, up_template;     name="up_input",   V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
    down_layer   = NeuronLayer(N, down_template;   name="down_input", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
    hidden_layer = NeuronLayer(N, hidden_template; name="hidden",     V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
    out_layer    = NeuronLayer(N, out_template;    name="output",     V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)

    up_to_hidden   = SynapseLayer(up_layer,   hidden_layer, exc_syn;
                    dist=NormalDist(0.5, 0.2), density=density, pre_idx=1, post_idx=3)
    down_to_hidden = SynapseLayer(down_layer, hidden_layer, exc_syn;
                    dist=NormalDist(0.5, 0.2), density=density, pre_idx=2, post_idx=3)
    hidden_to_out  = SynapseLayer(hidden_layer, out_layer,  exc_syn_out;
                    dist=NormalDist(0.5, 0.2), density=density, pre_idx=3, post_idx=4)

    inhib_up     = SynapseLayer(up_layer,     up_layer,     inhib_tmpl;
                    dist=UniformDist(0.05, inhib_str), density=inhib_den, pre_idx=1, post_idx=1)
    inhib_down   = SynapseLayer(down_layer,   down_layer,   inhib_tmpl;
                    dist=UniformDist(0.05, inhib_str), density=inhib_den, pre_idx=2, post_idx=2)
    inhib_hidden = SynapseLayer(hidden_layer, hidden_layer, inhib_tmpl;
                    dist=UniformDist(0.05, inhib_str), density=inhib_den, pre_idx=3, post_idx=3)
    inhib_out    = SynapseLayer(out_layer,    out_layer,    inhib_tmpl;
                    dist=UniformDist(0.05, inhib_str), density=inhib_den, pre_idx=4, post_idx=4)

    return LayeredNetwork(
        [up_layer, down_layer, hidden_layer, out_layer],
        [up_to_hidden, down_to_hidden, hidden_to_out,
         inhib_up, inhib_down, inhib_hidden, inhib_out]
    )
end

# ----------------------------------------------------------------------------------
# Hybrid construction — GENERIC over the two source records.
# The first half comes from `rec_a`, the second half from `rec_b`, regardless of
# their labels. So (healthy, infarction) gives an anomaly hybrid and
# (healthy, healthy) gives a control hybrid with the same stitching artefacts.
# ----------------------------------------------------------------------------------
function build_hybrid_spiketrain(rec_a, rec_b; transition_frac=0.5, fs=1000.0, Δ_val=Δ)
    st_a, _, _ = get_spiketrain(rec_a.patient, rec_a.session; Δ=Δ_val, fs=fs)
    st_b, _, _ = get_spiketrain(rec_b.patient, rec_b.session; Δ=Δ_val, fs=fs)

    total_duration = tsim
    transition_t   = total_duration * transition_frac
    transition_ms  = round(Int, transition_t * fs)

    a_part = filter(s -> s.time < transition_ms, st_a)
    b_part = [Spike(s.time + transition_ms, s.polarity, s.src_name)
              for s in st_b if s.time + transition_ms < total_duration * fs]

    hybrid = sort(vcat(a_part, b_part), by = x -> x.time)
    return hybrid, total_duration, transition_t
end

# -----------------------------------------------------------------------------
# Simulation - returns sampled output-weight trace + firing rates
# -----------------------------------------------------------------------------
function simulate_hybrid(spiketrain, total_duration; sample_interval=SAMPLE_INT,
                         freeze_at=Inf, builder=build_network, out_layer_idx=4,
                         pulse_amp_val=pulse_amp)
    nsteps     = Int(round(total_duration / dt))
    max_sample = total_duration * 1000.0
    spiketrain = filter(s -> s.time < max_sample, spiketrain)

    up_pulses   = zeros(nsteps)
    down_pulses = zeros(nsteps)
    for spike in spiketrain
        step = Int(floor(spike.time)) + 1
        1 <= step <= nsteps || continue
        spike.polarity ? (up_pulses[step] += pulse_amp_val) : (down_pulses[step] -= pulse_amp_val)
    end

    net = builder()
    input_fn = t -> begin
        idx = clamp(Int(floor(t / dt)) + 1, 1, nsteps)
        v = zeros(length(net.neuronlayers))
        v[1] = up_pulses[idx]
        v[2] = down_pulses[idx]
        v
    end
    n_out = net.neuronlayers[out_layer_idx].N

    sample_steps = Float64[]
    scores       = Float64[]
    firing_rates = Float64[]
    next_sample  = 0.0
    last_sample_t = -1.0

    cb = function(t, net, step)
        if t >= next_sample - 1e-9
            out_ws = Float64[]
            for syn in net.synapselayers
                if syn.post_idx == out_layer_idx && !syn.isinhibitory
                    push!(out_ws, mean(syn.ws))
                end
            end
            push!(sample_steps, t)
            push!(scores, mean(out_ws))
            out_layer = net.neuronlayers[out_layer_idx]
            n_fired = count(out_layer.t_lastout .> last_sample_t)
            push!(firing_rates, n_fired / n_out)
            last_sample_t = t
            next_sample += sample_interval
        end
    end

    runlayers!(net, dt, total_duration; inputfn=input_fn, callback=cb, freeze_at=freeze_at)
    return (times=sample_steps, scores=scores, firing_rates=firing_rates, net=net)
end

smooth(scores, window_size=5) =
    [mean(scores[max(1, i-div(window_size,2)):min(end, i+div(window_size,2))]) for i in eachindex(scores)]

# -------------------------------------------------------------------------
# Detection - fires an alarm on a sustained post-transition weight drop
# ----------------------------------------------------------------------------
function detect_anomaly(signal, times, transition_t; settle=SETTLE, min_consec_decrease=2, smooth_window=5)
    smoothed = smooth(signal, smooth_window)
    calib_mask = (times .> settle) .& (times .< transition_t)
    calib_idcs = findall(calib_mask)
    isempty(calib_idcs) && error("empty calibration window — check settle/transition vs tsim")

    alarm_idx = let
        consec_decrease = 0
        found = nothing
        for i in 2:length(smoothed)
            i <= calib_idcs[end] && continue
            if smoothed[i] < smoothed[i-1]
                consec_decrease += 1
            else
                consec_decrease = 0
            end
            if consec_decrease >= min_consec_decrease
                found = i
                break
            end
        end
        found
    end
    if alarm_idx !== nothing && times[alarm_idx] < transition_t
        alarm_idx = nothing
    end

    alarm_time    = alarm_idx !== nothing ? times[alarm_idx] : -1.0
    alarm_latency = alarm_time > 0 ? alarm_time - transition_t : -1.0

    early_mask = times .< transition_t
    late_mask  = times .>= transition_t
    return (alarm_idx=alarm_idx, alarm_time=alarm_time, alarm_latency=alarm_latency,
            smoothed=smoothed, calib_idcs=calib_idcs,
            early_mask=early_mask, late_mask=late_mask, settle=settle)
end

"""
    anomaly_score(det, times) -> Float64

Continuous, threshold-free anomaly score for ROC/PR analysis: the drop in mean
output weight from the calibration window to the post-transition window. Positive
and large ⇒ strong drift (anomaly); near zero ⇒ stable (healthy continuation).
"""
function anomaly_score(det, times)
    sm = det.smoothed
    calib = sm[det.calib_idcs]
    post  = sm[det.late_mask]
    (isempty(calib) || isempty(post)) && return 0.0
    return mean(calib) - mean(post)
end

"""
    run_trial(rec_a, rec_b) -> NamedTuple

One hybrid simulation. Returns the binary `detected` flag, the continuous
`score`, and the alarm latency.
"""
function run_trial(rec_a, rec_b; detect_on="weight")
    hybrid, total_t, transition_t = build_hybrid_spiketrain(rec_a, rec_b)
    res = simulate_hybrid(hybrid, total_t)
    sig = detect_on == "weight" ? res.scores : res.firing_rates
    det = detect_anomaly(sig, res.times, transition_t)
    return (detected = det.alarm_time > 0,
            score    = anomaly_score(det, res.times),
            latency  = det.alarm_latency,
            patient_a = rec_a.patient, patient_b = rec_b.patient)
end

# ------------------------------
# Build labelled trial set
# ------------------------------------------------------------
Random.seed!(SEED)
all_records = build_registry("./ecg-db")
labelled    = filter(r -> r.label != :unknown, all_records)
healthy     = filter(r -> r.label == :healthy,    labelled)
infarction  = filter(r -> r.label == :infarction, labelled)

length(healthy)    >= 2 || error("need at least 2 healthy records for the control pairs")
length(infarction) >= 1 || error("need at least 1 infarction record")

println("="^72)
println("CLASS DISTRIBUTION (database)")
println("="^72)
n_h, n_i = length(healthy), length(infarction)
println("  Healthy:    $n_h")
println("  Infarction: $n_i   (ratio 1 : $(round(n_i/n_h, digits=2)) healthy:infarction)")

# Positive trials: healthy → infarction
pos_pairs = [(sample(healthy), sample(infarction)) for _ in 1:N_POS]
# Negative (control) trials: healthy → DIFFERENT healthy
function distinct_healthy_pair()
    a = sample(healthy)
    b = sample(healthy)
    while b.patient == a.patient && b.session == a.session
        b = sample(healthy)
    end
    return (a, b)
end
neg_pairs = [distinct_healthy_pair() for _ in 1:N_NEG]

# ---------------
# Run trials
# ---------------------------
y_true = Bool[]   
y_pred = Bool[]  
scores = Float64[]
latencies = Float64[]
trial_kind = String[]

println("\nRunning $(N_POS) anomaly trials (healthy→infarction)…")
@showprogress for (a, b) in pos_pairs
    r = run_trial(a, b)
    push!(y_true, true); push!(y_pred, r.detected); push!(scores, r.score)
    push!(latencies, r.latency); push!(trial_kind, "anomaly")
end

println("Running $(N_NEG) control trials (healthy→healthy)…")
@showprogress for (a, b) in neg_pairs
    r = run_trial(a, b)
    push!(y_true, false); push!(y_pred, r.detected); push!(scores, r.score)
    push!(latencies, r.latency); push!(trial_kind, "control")
end

# ---------------
# Metrics
# --------------
rng = MersenneTwister(SEED)
S = summarize(y_true, y_pred; scores=scores, n_boot=N_BOOT, alpha=ALPHA, rng=rng)
c = S.confusion

pct(x) = round(100x, digits=1)
r3(x)  = round(x, digits=3)

println("\n" * "="^72)
println("RAW CONFUSION MATRIX  (positive = anomaly / alarm)")
println("="^72)
println("                     │  pred: ALARM   pred: no-alarm")
println("    actual: anomaly  │      $(lpad(c.tp,5))         $(lpad(c.fn,5))     (TP / FN)")
println("    actual: healthy  │      $(lpad(c.fp,5))         $(lpad(c.tn,5))     (FP / TN)")
println("    n = $(c.tp+c.fn+c.fp+c.tn)")

println("\n" * "="^72)
println("IMBALANCE-AWARE METRICS  (95% CI)")
println("="^72)
println("  Balanced accuracy :  $(pct(S.balanced_accuracy))%   [$(pct(S.balanced_acc_ci[1])), $(pct(S.balanced_acc_ci[2]))]")
println("  MCC               :  $(r3(S.mcc))     [$(r3(S.mcc_ci[1])), $(r3(S.mcc_ci[2]))]   (−1…1, 0=chance)")
println("  F1 (anomaly)      :  $(r3(S.f1))     [$(r3(S.f1_ci[1])), $(r3(S.f1_ci[2]))]")
println("  F2 (recall-heavy) :  $(r3(S.f2))")
println("  G-mean            :  $(r3(S.gmean))")
println("  Youden's J        :  $(r3(S.youden_j))")
println("  Cohen's κ         :  $(r3(S.cohens_kappa))")
println("  Accuracy (ref)    :  $(pct(S.accuracy))%   ← misleading under imbalance, shown for context")

println("\n" * "="^72)
println("PER-CLASS PRECISION / RECALL")
println("="^72)
println("  class      precision   recall   support")
println("  anomaly      $(lpad(r3(S.precision_anomaly),6))   $(lpad(r3(S.recall_anomaly),6))    $(lpad(S.per_class.anomaly.support,4))")
println("  healthy      $(lpad(r3(S.precision_healthy),6))   $(lpad(r3(S.recall_healthy),6))    $(lpad(S.per_class.healthy.support,4))")
println("  (anomaly recall = sensitivity; healthy recall = specificity)")

println("\n" * "="^72)
println("RATES WITH WILSON CIs")
println("="^72)
println("  Sensitivity (TPR) :  $(pct(S.sensitivity))%   [$(pct(S.sensitivity_ci[1])), $(pct(S.sensitivity_ci[2]))]")
println("  Specificity (TNR) :  $(pct(S.specificity))%   [$(pct(S.specificity_ci[1])), $(pct(S.specificity_ci[2]))]")
println("  NPV               :  $(pct(S.npv))%")

println("\n" * "="^72)
println("THRESHOLD-FREE RANKING METRICS")
println("="^72)
println("  AUROC             :  $(r3(S.auroc))   (0.5 = chance)")
println("  Average precision :  $(r3(S.average_precision))   (no-skill = prevalence = $(r3(count(y_true)/length(y_true))))")

# -----------------------------
# HEALTHY -> HEALTHY CONTROL
# ----------------------------
control_alarms = count(y_pred[i] for i in eachindex(y_pred) if !y_true[i])
control_total  = count(!, y_true)
far = control_alarms / control_total
far_ci = wilson_ci(control_alarms, control_total)

println("\n" * "="^72)
println("HEALTHY → HEALTHY CONTROL  (ideally NO anomaly detected)")
println("="^72)
println("  Trials               :  $control_total")
println("  False alarms         :  $control_alarms")
println("  False-alarm rate     :  $(pct(far))%   95% CI [$(pct(far_ci[1])), $(pct(far_ci[2]))]")
println("  → Correct rejections :  $(pct(1 - far))%   (network stays silent on a healthy→healthy stitch)")
if control_alarms > 0
    println("\n  Patients that triggered a false alarm:")
    for i in eachindex(y_pred)
        (!y_true[i] && y_pred[i]) || continue
        # find original pair index within the negative block
        ni = i - N_POS
        a, b = neg_pairs[ni]
        println("    • $(a.patient)/$(a.session) → $(b.patient)/$(b.session)  (score=$(r3(scores[i])))")
    end
end

# Distribution separation between control vs anomaly scores
ctrl_scores = scores[.!y_true]
anom_scores = scores[y_true]
println("\n  Anomaly-score separation:")
println("    control μ = $(r3(mean(ctrl_scores)))  ±$(r3(std(ctrl_scores)))")
println("    anomaly μ = $(r3(mean(anom_scores)))  ±$(r3(std(anom_scores)))")

# --------
# Plots 
# --------
try
    imgdir = joinpath(@__DIR__, "../docs/imgs")
    isdir(imgdir) || mkpath(imgdir)

    # Confusion-matrix heatmap
    M = S.matrix
    ph = heatmap(1:2, 1:2, M;
                 c=:blues, title="Confusion matrix (n=$(length(y_true)))",
                 xlabel="predicted", ylabel="actual", aspect_ratio=1, yflip=true,
                 xticks=(1:2, ["ALARM", "no-alarm"]),
                 yticks=(1:2, ["anomaly", "healthy"]))
    for i in 1:2, j in 1:2
        annotate!(ph, [(j, i, text(string(M[i, j]), 12, :black))])
    end
    savefig(ph, joinpath(imgdir, "eval_confusion_matrix.png"))

    # ROC curve
    thr = sort(unique(scores); rev=true)
    tprs = Float64[]; fprs = Float64[]
    for t in vcat(Inf, thr, -Inf)
        pred = scores .>= t
        cc = confusion_counts(y_true, pred)
        push!(tprs, recall_score(cc))
        push!(fprs, cc.fp + cc.tn == 0 ? 0.0 : cc.fp / (cc.fp + cc.tn))
    end
    pr = plot(fprs, tprs; lw=2, label="ROC (AUROC=$(r3(S.auroc)))",
              xlabel="False-positive rate", ylabel="True-positive rate",
              title="ROC — anomaly vs healthy control", legend=:bottomright)
    plot!(pr, [0,1], [0,1]; ls=:dash, color=:gray, label="chance")
    savefig(pr, joinpath(imgdir, "eval_roc.png"))

    # Score distributions
    pd = histogram(ctrl_scores; bins=20, alpha=0.5, label="control (healthy→healthy)",
                   color=:blue, normalize=true, xlabel="anomaly score (weight drop)",
                   ylabel="density", title="Anomaly-score distributions")
    histogram!(pd, anom_scores; bins=20, alpha=0.5, label="anomaly (healthy→infarction)",
               color=:red, normalize=true)
    vline!(pd, [0.0]; color=:black, ls=:dot, label="no drift")
    savefig(pd, joinpath(imgdir, "eval_score_distributions.png"))

    println("\n  Plots saved to docs/imgs/: eval_confusion_matrix.png, eval_roc.png, eval_score_distributions.png")
catch err
    println("\n  [plotting skipped: $(err)]")
end

println("\n" * "="^72)
println("DONE")
println("="^72)
