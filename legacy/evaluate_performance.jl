include("../modules/Layers.jl")
include("../modules/Signals.jl")
include("../modules/Registry.jl")
include("../modules/Metrics.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils, .Registry, .Metrics
using Statistics, Random, Plots, StatsPlots, ProgressMeter
using StatsBase: sample

plotly()

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

db_root     = "./ecg-db"
all_records = build_registry(db_root)
labelled    = filter(r -> r.label != :unknown, all_records)

Random.seed!(SEED)
healthy_records    = filter(r -> r.label == :healthy,    labelled)
infarction_records = filter(r -> r.label == :infarction, labelled)

n_healthy = length(healthy_records)
n_infarction = length(infarction_records)
n_total = n_healthy + n_infarction

println("="^70)
println("CLASS DISTRIBUTION")
println("="^70)
println("  Healthy:    $n_healthy ($(round(100 * n_healthy / n_total, digits=1))%)")
println("  Infarction: $n_infarction ($(round(100 * n_infarction / n_total, digits=1))%)")
println("  Ratio:      1 : $(round(n_infarction / n_healthy, digits=2)) (infarction:healthy)")
println("  Majority-class baseline accuracy: $(round(100 * n_infarction / n_total, digits=1))%")
println("  Balanced baseline accuracy:       50.0%")

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
                    dist=NormalDist(0.5, 0.2), density=density,       pre_idx=1, post_idx=3)
    down_to_hidden = SynapseLayer(down_layer, hidden_layer, exc_syn;
                    dist=NormalDist(0.5, 0.2), density=density,       pre_idx=2, post_idx=3)
    hidden_to_out  = SynapseLayer(hidden_layer, out_layer,  exc_syn_out;
                    dist=NormalDist(0.5, 0.2), density=density,       pre_idx=3, post_idx=4)

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

function build_hybrid_spiketrain(healthy_rec, infarction_rec; transition_frac=0.5, fs=1000.0, Δ_val=Δ)
    st_healthy, _, _ = get_spiketrain(healthy_rec.patient, healthy_rec.session; Δ=Δ_val, fs=fs)
    st_infarction, _, _ = get_spiketrain(infarction_rec.patient, infarction_rec.session; Δ=Δ_val, fs=fs)

    total_duration = tsim
    transition_t   = total_duration * transition_frac
    transition_ms  = round(Int, transition_t * fs)

    healthy_part = filter(s -> s.time < transition_ms, st_healthy)
    infarction_part = [Spike(s.time + transition_ms, s.polarity, s.src_name)
                       for s in st_infarction if s.time + transition_ms < total_duration * fs]

    hybrid = sort(vcat(healthy_part, infarction_part), by = x -> x.time)
    return hybrid, total_duration, transition_t
end

function simulate_hybrid(spiketrain, total_duration; sample_interval=0.2, freeze_at=Inf,
                          builder=build_network, out_layer_idx=4, pulse_amp_val=pulse_amp)
    nsteps       = Int(round(total_duration / dt))
    max_sample   = total_duration * 1000.0

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
        n_neur = length(net.neuronlayers)
        v = zeros(n_neur)
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
            w_mean = mean(out_ws)
            push!(sample_steps, t)
            push!(scores, w_mean)
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

function smooth(scores, window_size=5)
    half = div(window_size, 2)
    [mean(scores[max(1, i-half):min(end, i+half)]) for i in eachindex(scores)]
end

function detect_anomaly(signal, times, transition_t; settle=5.0, min_consec_decrease=2, smooth_window=5)
    smoothed = smooth(signal, smooth_window)
    calib_mask = (times .> settle) .& (times .< transition_t)
    calib_idcs = findall(calib_mask)
    n_calib = length(calib_idcs)

    alarm_idx = let
        consec_decrease = 0
        found = nothing
        for i in 2:length(smoothed)
            if i <= calib_idcs[end]
                continue
            end
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

    alarm_time = alarm_idx !== nothing ? times[alarm_idx] : -1.0
    alarm_latency = alarm_time > 0 ? alarm_time - transition_t : -1.0

    early_mask = times .< transition_t
    late_mask  = times .>= transition_t
    healthy_seg   = signal[early_mask]
    infarction_seg = signal[late_mask]

    if alarm_idx !== nothing
        hp_above = count(early_mask[alarm_idx:end])
        ip_above = count(late_mask[alarm_idx:end])
    else
        hp_above = 0
        ip_above = 0
    end
    fp_rate = length(healthy_seg) > 0 ? hp_above / length(healthy_seg) : 0.0
    det_rate = length(infarction_seg) > 0 ? ip_above / length(infarction_seg) : 0.0

    return (alarm_idx=alarm_idx, alarm_time=alarm_time, alarm_latency=alarm_latency,
            healthy_seg=healthy_seg, infarction_seg=infarction_seg,
            fp_rate=fp_rate, det_rate=det_rate,
            early_mask=early_mask, late_mask=late_mask,
            calib_idcs=calib_idcs, n_calib=n_calib,
            smoothed=smoothed, min_consec_decrease=min_consec_decrease,
            settle=settle)
end

function run_hybrid_test(healthy_rec, infarction_rec;
                         transition_frac=0.5, sample_interval=0.2,
                         freeze_at=Inf, detect_on="weight",
                         builder=build_network, out_layer_idx=4,
                         pulse_amp_val=pulse_amp, Δ_val=Δ)
    hybrid_train, hybrid_len, transition_t = build_hybrid_spiketrain(
        healthy_rec, infarction_rec; transition_frac=transition_frac, Δ_val=Δ_val
    )

    result = simulate_hybrid(hybrid_train, hybrid_len;
                             sample_interval=sample_interval, freeze_at=freeze_at,
                             builder=builder, out_layer_idx=out_layer_idx,
                             pulse_amp_val=pulse_amp_val)

    signal = detect_on == "weight" ? result.scores : result.firing_rates
    det = detect_anomaly(signal, result.times, transition_t)

    return merge((result=result, transition_t=transition_t,
                  patient_healthy=healthy_rec.patient,
                  patient_infarction=infarction_rec.patient,
                  hybrid_len=hybrid_len), det)
end

# -----------------------------------------------------------------------------------
Random.seed!(SEED)
demo_healthy    = sample(healthy_records)
demo_infarction = sample(infarction_records)

println("\n" * "="^70)
println("SINGLE PAIR DEMO")
println("="^70)
println("  Healthy:    $(demo_healthy.patient)")
println("  Infarction: $(demo_infarction.patient)")

demo_result = run_hybrid_test(demo_healthy, demo_infarction; detect_on="weight")

wt = demo_result.result.scores
t  = demo_result.result.times
tt = demo_result.transition_t

st = demo_result.smoothed
wt_lo = minimum(wt) - 0.05
wt_hi = maximum(wt) + 0.05

demo_ba = balanced_acc(demo_result.det_rate, 1.0 - demo_result.fp_rate)
demo_kl = kl_estimate(demo_result.healthy_seg, demo_result.infarction_seg)
# demo_js = js_divergence_estimate(demo_result.healthy_seg, demo_result.infarction_seg)

println("\n  Detection:  $(demo_result.alarm_time > 0 ? "YES" : "NO") | " *
        "Latency: $(demo_result.alarm_time > 0 ? "$(round(demo_result.alarm_latency, digits=1))s" : "N/A")")
println("  FP rate:    $(round(100 * demo_result.fp_rate, digits=1))%")
println("  Det rate:   $(round(100 * demo_result.det_rate, digits=1))%")
println("  Balanced accuracy: $(round(100 * demo_ba, digits=1))%")
println("  Healthy μ: $(round(mean(demo_result.healthy_seg), digits=4))  |  " *
        "Infarction μ: $(round(mean(demo_result.infarction_seg), digits=4))")
println("  KL(healthy ‖ infarction): $(round(demo_kl, digits=4))")
# println("  JS(healthy, infarction):  $(round(demo_js, digits=4))")

p1 = plot(t, wt; label="Raw weight mean", linewidth=1, color=:purple, alpha=0.4,
          xlabel="Time (s)", ylabel="mean output weight",
          title="Hybrid Demo: $(demo_result.patient_healthy) → $(demo_result.patient_infarction)",
          legend=:topleft)
plot!(p1, t, st; label="Smoothed (7-point)", linewidth=2.5, color=:darkviolet)
plot!(p1, [demo_result.settle, tt], [wt_lo, wt_lo];
      fillrange=[wt_hi, wt_hi], alpha=0.12, color=:blue, label="Calibration")
plot!(p1, [0, demo_result.settle], [wt_lo, wt_lo];
      fillrange=[wt_hi, wt_hi], alpha=0.06, color=:gray, label="Convergence")
plot!(p1, [tt, demo_result.hybrid_len], [wt_lo, wt_lo];
      fillrange=[wt_hi, wt_hi], alpha=0.12, color=:red, label="Infarction")
vline!(p1, [tt], color=:black, line=:dash, linewidth=2, label="Transition")

diffs = diff(st)
p2 = plot(t[2:end], diffs; label="Δ smoothed weight", linewidth=2, color=:teal,
          xlabel="Time (s)", ylabel="Δ weight / sample",
          title="Weight Derivative — negative = decreasing trend")
hline!(p2, [0], color=:black, line=:dot, linewidth=1, label="Zero")
if demo_result.alarm_time > 0
    vline!(p2, [demo_result.alarm_time], color=:orange, line=:solid, linewidth=2,
           label="Alarm ($(demo_result.min_consec_decrease) consecutive ↓)")
    annotate!(p1, [(demo_result.alarm_time, wt_hi - 0.02,
                    "⬆ $(round(demo_result.alarm_latency, digits=1))s", :orange)])
end

p_demo = plot(p1, p2, layout=(2, 1), size=(1000, 600))
savefig(p_demo, joinpath(@__DIR__, "../docs/imgs/evaluation_demo.png"))
display(p_demo)
println("  Plot saved to docs/imgs/evaluation_demo.png")

n_pairs = min(50, n_healthy, n_infarction)
println("\n" * "="^70)
println("MULTI-PAIR EVALUATION ($n_pairs random pairs)")
println("="^70)

pairs = [(sample(healthy_records), sample(infarction_records)) for _ in 1:n_pairs]

detected   = Bool[]
latencies  = Float64[]
fp_rates   = Float64[]
det_rates  = Float64[]
bal_accs   = Float64[]
kl_divs    = Float64[]
js_divs    = Float64[]
h_means    = Float64[]
i_means    = Float64[]

@showprogress for (i, (h_rec, i_rec)) in enumerate(pairs)
    r = run_hybrid_test(h_rec, i_rec; detect_on="weight")

    ba = balanced_acc(r.det_rate, 1.0 - r.fp_rate)
    kl = kl_estimate(r.healthy_seg, r.infarction_seg)
    # js = js_divergence_estimate(r.healthy_seg, r.infarction_seg)

    push!(detected,  r.alarm_time > 0)
    push!(latencies, r.alarm_latency)
    push!(fp_rates,  r.fp_rate)
    push!(det_rates, r.det_rate)
    push!(bal_accs,  ba)
    push!(kl_divs,   kl)
    # push!(js_divs,   js)
    push!(h_means,   mean(r.healthy_seg))
    push!(i_means,   mean(r.infarction_seg))
end

# ── Aggregate ──
det_rate_agg = mean(detected)
mean_latency = mean(latencies[detected])
mean_fp      = mean(fp_rates)
mean_det     = mean(det_rates)
mean_ba      = mean(bal_accs)
mean_kl      = mean(kl_divs)
# mean_js      = mean(js_divs)
mean_h       = mean(h_means)
mean_i       = mean(i_means)
std_ba       = std(bal_accs)
std_kl       = std(kl_divs)

# ── Distribution comparison per-pair ──
println("\n" * "─"^70)
println("PER-CLASS PERFORMANCE")
println("─"^70)
println("  Detection rate (alarm fired):        $(round(100 * det_rate_agg, digits=1))%")
println("  Avg latency (when detected):         $(round(mean_latency, digits=2))s ± $(round(std(latencies[detected]), digits=2))")
println("  Avg true positive rate (det_rate):   $(round(100 * mean_det, digits=1))%")
println("  Avg false positive rate:             $(round(100 * mean_fp, digits=1))%")
println("  Avg true negative rate (1 - fp_rate): $(round(100 * (1 - mean_fp), digits=1))%")

println("\n" * "─"^70)
println("BALANCE-AWARE METRICS")
println("─"^70)
println("  Balanced accuracy:                   $(round(100 * mean_ba, digits=1))% ± $(round(100 * std_ba, digits=1))")
println("    (Individual: ", join(["$(round(100 * b, digits=1))" for b in bal_accs], ", "), ")")
println("    → Chance level: 50% | Majority-class baseline: $(round(100 * n_infarction / n_total, digits=1))%")

println("\n" * "─"^70)
println("DISTRIBUTION SEPARATION METRICS")
println("─"^70)
println("  KL divergence (healthy ‖ infarction): $(round(mean_kl, digits=4)) ± $(round(std_kl, digits=4))")
println("    → 0 = identical distributions, higher = more separable")
# println("  JS divergence (healthy, infarction):  $(round(mean_js, digits=4))")
println("    → Bounded [0, $(round(log(2), digits=4))], symmetric")
println("  Healthy score μ:                      $(round(mean_h, digits=4))")
println("  Infarction score μ:                   $(round(mean_i, digits=4))")
println("  Score separation (|Δμ| / |μ_h|):       $(round(abs(mean_i - mean_h) / (abs(mean_h) + 1e-8), digits=4))")

# ── Score distribution histogram ──
all_healthy_scores = Float64[]
all_infarction_scores = Float64[]
for (h_rec, i_rec) in pairs
    r = run_hybrid_test(h_rec, i_rec; detect_on="weight")
    append!(all_healthy_scores, r.healthy_seg)
    append!(all_infarction_scores, r.infarction_seg)
    if length(all_healthy_scores) > 10000
        break
    end
end

p3 = histogram(all_healthy_scores; alpha=0.5, bins=40, label="Healthy",
               color=:blue, normalize=true,
               xlabel="Mean output weight", ylabel="Density",
               title="Pooled Score Distribution ($(length(pairs)) pairs)")
histogram!(p3, all_infarction_scores; alpha=0.5, bins=40, label="Infarction",
           color=:red, normalize=true)
savefig(p3, joinpath(@__DIR__, "../docs/imgs/evaluation_distributions.png"))
display(p3)
println("\n  Histogram saved to docs/imgs/evaluation_distributions.png")

println("\n" * "="^70)
println("EVALUATION SUMMARY")
println("="^70)
println("  Balanced Accuracy:  $(round(100 * mean_ba, digits=1))%  (unbiased by class ratio 1:$(round(n_infarction / n_healthy, digits=1)))")
println("  KL Divergence:      $(round(mean_kl, digits=4))")
println("  Detection Rate:     $(round(100 * det_rate_agg, digits=1))%  (alarm fired)")
println("  Avg FP Rate:        $(round(100 * mean_fp, digits=1))%")
println("  Avg Detection Rate: $(round(100 * mean_det, digits=1))%")
println("="^70)
