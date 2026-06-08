include("../modules/Layers.jl")
include("../modules/Signals.jl")
include("../modules/Registry.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils, .Registry
using Statistics, Random, Plots, StatsPlots, ProgressMeter
using StatsBase: sample

plotly()

# ----- Final params from paramsearch,jl -----
const Δ = 0.07
const pulse_amp = 162.19
const N = 25
const τ_m_in = 5.96
const τ_m_out = 8.95
const τ_s_in = 27.71
const τ_s_out = 18.27
const R_m_in = 5.32
const R_m_out = 1.44
const τ_ref_in = 2.5
const τ_ref_out = 1.32
const τ_pretrace = 9.47
const τ_posttrace  = 57.41
const learningrate = 0.11
const learningrate2 = 0.13
const density = 0.89
const inhib_density = 0.6
const inhib_str = 0.32

const dt = 0.001
const tsim = 25.0
const SEED = 67

# ----- Registry -----
dbroot = "./ecg-db"
all_records = build_registry(dbroot)
labelled = filter(r -> r.label != :unknown, all_records)
healthy = filter(r -> r.label == :healthy, labelled)
infarction = filter(r -> r.label == :infarction, labelled)

# ----- Building the network -----
function buildnet()
    Random.seed!(SEED)

    # Neuron templates
    up_template = Neuron("up"; R_m=R_m_in, τ_m=τ_m_in, τ_s=τ_s_in, 
                    τ_ref=τ_ref_in, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    down_template = Neuron("down"; R_m=R_m_in, τ_m=τ_m_in, τ_s=τ_s_in, 
                    τ_ref=τ_ref_in, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace, isreverse=true)
    hid_template = Neuron("hidden"; R_m=R_m_out, τ_m=τ_m_out, τ_s=τ_s_out, 
                    τ_ref=τ_ref_out, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    out_template = Neuron("out"; R_m=R_m_out, τ_m=τ_m_out, τ_s=τ_s_out, 
                    τ_ref=τ_ref_out, τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)

    # Synapse template
    syn_template = Synapse(1, 2; learningrate=learningrate, wmax=1.0)
    syn_template_out = Synapse(2, 3; learningrate=learningrate2, wmax=1.0)
    syn_inhib_template = Synapse(1, 1; learningrate=0.0, wmax=1.0, isinhibitory=true)

    # Neuron layers
    up_layer = NeuronLayer(N, up_template; name="up", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
    down_layer = NeuronLayer(N, down_template; name="down", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
    hid_layer = NeuronLayer(N, hid_template; name="hidden", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
    out_layer = NeuronLayer(N, out_template; name="out", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)

    # Synapse layers
    up_to_hid = SynapseLayer(up_layer, hid_layer, syn_template; pre_idx=1, post_idx=3, 
                    dist=NormalDist(0.5, 0.2), density=density)
    down_to_hid = SynapseLayer(down_layer, hid_layer, syn_template; pre_idx=2, post_idx=3,
                    dist=NormalDist(0.5, 0.2), density=density)
    hid_to_out = SynapseLayer(hid_layer, out_layer, syn_template_out; pre_idx=3, post_idx=4,
                    dist=NormalDist(0.5, 0.2), density=density)

    # Lateral inhibition layers
    inhib_up = SynapseLayer(up_layer, up_layer, syn_inhib_template; pre_idx=1, post_idx=1,
                    dist=UniformDist(0.05, inhib_str), density=inhib_density)
    inhib_down = SynapseLayer(down_layer, down_layer, syn_inhib_template; pre_idx=2, post_idx=2,
                    dist=UniformDist(0.05, inhib_str), density=inhib_density)
    inhib_hid = SynapseLayer(hid_layer, hid_layer, syn_inhib_template; pre_idx=3, post_idx=3,
                    dist=UniformDist(0.05, inhib_str), density=inhib_density)
    inhib_out = SynapseLayer(out_layer, out_layer, syn_inhib_template; pre_idx=4, post_idx=4,
                    dist=UniformDist(0.05, inhib_str), density=inhib_density)

    return LayeredNetwork(
        [up_layer, down_layer, hid_layer, out_layer],
        [up_to_hid, down_to_hid, hid_to_out, 
        inhib_up, inhib_down, inhib_hid, inhib_out]
    )
end


# ----- Hybrid spiketrain construction -----
function get_hybrid_spiketrain(healthy, infarction; transition=0.5, fs=1000.0, Δ=Δ)
    st_healthy, _, _ = get_spiketrain(healthy.patient, healthy.session; Δ=Δ, fs=fs)
    st_infarction, _, _ = get_spiketrain(infarction.patient, infarction.session, Δ=Δ, fs=fs)

    total_t = tsim
    transition_t = tsim * transition
    transition_ms = round(Int, transition_t * fs)

    healthy_part = filter(s -> s.time < transition_ms, st_healthy)
    infarction_part = [Spike(s.time + transition_ms, s.polarity, s.src_name)
                       for s in st_infarction if s.time + transition_ms < total_t * fs]

    hybrid = sort(vcat(healthy_part, infarction_part), by = x -> x.time)

    return hybrid, total_t, transition_t
end

# ----- Detection results -----
struct HybridDetectionResult
    patient_healthy::String
    patient_infarction::String
    detected::Bool
    alarm_time::Float64
    latency::Float64
    fp_rate::Float64
    det_rate::Float64
    healthy_mean::Float64
    infarction_mean::Float64
    healthy_n::Int
    infarction_n::Int
    n_calib::Int
end


# ----- Simulation -----
function simulate_hybrid(spiketrain, total_t; sample_interval=0.2, freeze_at=Inf,
                          builder=buildnet, out_layer_idx=4, pulse_amp_val=pulse_amp)
    nsteps = Int(round(total_t / dt))
    max_sample = total_t * 1000.0

    spiketrain = filter(s -> s.time < max_sample, spiketrain)

    up_pulses = zeros(nsteps)
    down_pulses = zeros(nsteps)

    for spike in spiketrain
        step = Int(floor(spike.time)) + 1
        1 ≤ step ≤ nsteps || continue
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
    scores = Float64[]
    firing_rates = Float64[]
    next_sample = 0.0
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

    runlayers!(net, dt, total_t; inputfn=input_fn, callback=cb, freeze_at=freeze_at)

    return (times=sample_steps, scores=scores, firing_rates=firing_rates, net=net)
end

# ----- Smoothing -----
function smooth(scores, window_size=5)
    half = div(window_size, 2)
    [mean(scores[max(1, i-half):min(end, i+half)]) for i in eachindex(scores)]
end


# ----- Detection -----
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


# ----- Run one hybrid test pair -----
function run_hybrid_test(healthy_rec, infarction_rec;
                         transition_frac=0.5, sample_interval=0.2,
                         freeze_at=Inf, detect_on="weight",
                         builder=buildnet, out_layer_idx=4,
                         pulse_amp_val=pulse_amp, Δ_val=Δ)
    hybrid_train, hybrid_len, transition_t = get_hybrid_spiketrain(
        healthy_rec, infarction_rec; transition=transition_frac, Δ=Δ_val
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

# ----- Results -----
Random.seed!(SEED)
demo_healthy    = sample(healthy)
demo_infarction = sample(infarction)

println("\n" * "="^70)
println("HYBRID SIGNAL — SINGLE CONTINUOUS SIMULATION (DEMO)")
println("="^70)
println("  Healthy:    $(demo_healthy.patient)")
println("  Infarction: $(demo_infarction.patient)")

demo_result = run_hybrid_test(demo_healthy, demo_infarction; detect_on="weight")

wt = demo_result.result.scores
t  = demo_result.result.times
tt = demo_result.transition_t
hl = demo_result.hybrid_len

st = demo_result.smoothed
wt_lo = minimum(wt) - 0.05
wt_hi = maximum(wt) + 0.05
println("\n" * "-"^70)
println("DEMO RESULT")
println("-"^70)
println("  Transition: $(round(tt, digits=1))s | Detected: $(demo_result.alarm_time > 0 ? "✅ at $(round(demo_result.alarm_time, digits=1))s" : "❌")")
if demo_result.alarm_time > 0
    println("  Latency: $(round(demo_result.alarm_latency, digits=1))s")
end
println("  Healthy μ: $(round(mean(demo_result.healthy_seg), digits=4))  |  Infarction μ: $(round(mean(demo_result.infarction_seg), digits=4))")
println("  FP: $(round(100 * demo_result.fp_rate, digits=1))%  |  Detection: $(round(100 * demo_result.det_rate, digits=1))%")