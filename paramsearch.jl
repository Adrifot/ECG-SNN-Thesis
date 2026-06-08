include("./modules/Layers.jl")
include("./modules/Signals.jl")
include("./modules/Registry.jl")

using .Signals
using .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils
using .Registry

using Random
using ProgressMeter
using Statistics
using StatsBase: sample

# Search consts
const SEARCH_ITERATIONS = 500
const FOCUSED_ITERATIONS = 50
const FOCUSED_TOPK = 5
const VALIDATION_PAIRS = 100
const TRANSITION_FRAC = 0.6
const SAMPLE_INTERVAL = 0.2
const HOLDOUT_FRAC = 0.4

const dt = 0.001
const tsim = 25.0
const SEED = 67

# ----- Registry & train/test split -----
db_root = "./ecg-db"
all_records = build_registry(db_root)
labelled = filter(r -> r.label != :unknown, all_records)

Random.seed!(SEED)
healthy_records = filter(r -> r.label == :healthy, labelled)
infarction_records = filter(r -> r.label == :infarction, labelled)

n_healthy_holdout = max(1, round(Int, length(healthy_records) * HOLDOUT_FRAC))
n_infarction_holdout = max(1, round(Int, length(infarction_records) * HOLDOUT_FRAC))

healthy_holdout = sample(healthy_records, n_healthy_holdout; replace=false)
infarction_holdout = sample(infarction_records, n_infarction_holdout; replace=false)

healthy_train = setdiff(healthy_records, healthy_holdout)
infarction_train = setdiff(infarction_records, infarction_holdout)

println("Available: $(length(healthy_records)) healthy, $(length(infarction_records)) infarction")
println("Train:     $(length(healthy_train)) healthy, $(length(infarction_train)) infarction")
println("Holdout:   $(length(healthy_holdout)) healthy, $(length(infarction_holdout)) infarction")

# ----- HyperParams -----
struct HyperParams
    Δ::Float64
    pulse_amp::Float64
    ltp_rate::Float64
    ltp_rate_out::Float64
    τ_s::Float64
    τ_s_output::Float64
    τ_pretrace::Float64
    τ_posttrace::Float64
    R_m_input::Float64
    R_m_output::Float64
    τ_m_input::Float64
    τ_m_output::Float64
    τ_ref_input::Float64
    τ_ref_output::Float64
    N::Int
    density::Float64
    inhib_den::Float64
    inhib_str::Float64
end

# ----- Deterministic hash for per-config seeding -----
function param_hash(hp::HyperParams, salt::Int=0)
    h = hash((hp.Δ, hp.pulse_amp, hp.ltp_rate, hp.ltp_rate_out,
              hp.τ_s, hp.τ_s_output, hp.τ_pretrace, hp.τ_posttrace,
              hp.R_m_input, hp.R_m_output, hp.τ_m_input, hp.τ_m_output,
              hp.τ_ref_input, hp.τ_ref_output, hp.N, hp.density,
              hp.inhib_den, hp.inhib_str))
    return mod(h + salt, typemax(Int32))
end

# ----- Distribution helpers -----
getrnd(a, b, rng) = rand(rng) * (b - a) + a

function sample_params_random(rng)
    HyperParams(
        getrnd(0.05, 0.2, rng), # Δ
        getrnd(100.0, 300.0, rng), # pulse_amp
        getrnd(0.1, 0.25, rng), # ltp_rate
        getrnd(0.1, 0.25, rng), # ltp_rate_out
        getrnd(25.0, 50.0, rng), # τ_s
        getrnd(10.0, 40.0, rng), # τ_s_output
        getrnd(8.0, 10.0, rng), # τ_pretrace
        getrnd(30.0, 60.0, rng), # τ_posttrace
        getrnd(5.0, 8.0, rng), # R_m_input
        getrnd(0.5, 1.5, rng), # R_m_output
        getrnd(5.0, 10.0, rng), # τ_m_input
        getrnd(2.0, 9.0, rng), # τ_m_output
        getrnd(1.0, 8.0, rng), # τ_ref_input
        getrnd(1.0, 3.0, rng), # τ_ref_output
        rand(rng, [15, 20, 25, 30]), # N
        getrnd(0.6, 0.9, rng), # density
        getrnd(0.4, 0.8, rng), # inhib_den
        getrnd(0.1, 0.5, rng) # inhib_str
    )
end

function sample_params_focused(rng, best::HyperParams; frac=0.15)
    function perturb(val, lo, hi)
        clamp(val * (1 + (2*rand(rng) - 1) * frac), lo, hi)
    end
    HyperParams(
        perturb(best.Δ, 0.01, 0.25),
        perturb(best.pulse_amp, 10.0, 100.0),
        perturb(best.ltp_rate, 0.05, 0.25),
        perturb(best.ltp_rate_out, 0.05, 0.25),
        perturb(best.τ_s, 1.0, 50.0),
        perturb(best.τ_s_output, 1.0, 50.0),
        perturb(best.τ_pretrace, 5.0, 30.0),
        perturb(best.τ_posttrace, 10.0, 60.0),
        perturb(best.R_m_input, 0.1, 10.0),
        perturb(best.R_m_output, 0.1, 10.0),
        perturb(best.τ_m_input, 0.5, 10.0),
        perturb(best.τ_m_output, 0.1, 10.0),
        perturb(best.τ_ref_input, 0.5, 10.0),
        perturb(best.τ_ref_output, 0.5, 10.0),
        best.N,
        perturb(best.density, 0.5, 0.99),
        perturb(best.inhib_den, 0.1, 0.9),
        perturb(best.inhib_str, 0.1, 0.9)
    )
end

# ----- 3.5 Layer Network -----
function build_network(hp::HyperParams, rng::AbstractRNG)
    # Neuron templates
    up_template = Neuron("up_input";
                    R_m=hp.R_m_input, τ_m=hp.τ_m_input, τ_s=hp.τ_s, τ_ref=hp.τ_ref_input,
                    τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace)
    down_template = Neuron("down_input";
                    R_m=hp.R_m_input, τ_m=hp.τ_m_input, τ_s=hp.τ_s, τ_ref=hp.τ_ref_input,
                    τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace, isreverse=true)
    hidden_template = Neuron("hidden";
                    R_m=hp.R_m_output, τ_m=hp.τ_m_output, τ_s=hp.τ_s_output, τ_ref=hp.τ_ref_output,
                    τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace)
    out_template = Neuron("output";
                    R_m=hp.R_m_output, τ_m=hp.τ_m_output, τ_s=hp.τ_s_output, τ_ref=hp.τ_ref_output,
                    τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace)

    # Synapse templates
    exc_syn = Synapse(1, 2; learningrate=hp.ltp_rate, wmax=1.0)
    exc_syn_out = Synapse(1, 2; learningrate=hp.ltp_rate_out, wmax=1.0)
    inhib_tmpl = Synapse(1, 1; learningrate=0.0, wmax=1.0, isinhibitory=true)

    # Neuron layers
    up_layer = NeuronLayer(hp.N, up_template; name="up_input", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15, rng=rng)
    down_layer = NeuronLayer(hp.N, down_template; name="down_input", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15, rng=rng)
    hidden_layer = NeuronLayer(hp.N, hidden_template; name="hidden", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15, rng=rng)
    out_layer = NeuronLayer(hp.N, out_template; name="output", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15, rng=rng)

    # Synapse layers
    up_to_hidden = SynapseLayer(up_layer, hidden_layer, exc_syn;
                    dist=NormalDist(0.5, 0.2), density=hp.density, pre_idx=1, post_idx=3, rng=rng)
    down_to_hidden = SynapseLayer(down_layer, hidden_layer, exc_syn;
                    dist=NormalDist(0.5, 0.2), density=hp.density, pre_idx=2, post_idx=3, rng=rng)
    hidden_to_out = SynapseLayer(hidden_layer, out_layer, exc_syn_out;
                    dist=NormalDist(0.5, 0.2), density=hp.density, pre_idx=3, post_idx=4, rng=rng)

    # Lateral inhibition layers
    inhib_up = SynapseLayer(up_layer, up_layer, inhib_tmpl;
                    dist=UniformDist(0.05, hp.inhib_str), density=hp.inhib_den, pre_idx=1, post_idx=1, rng=rng)
    inhib_down = SynapseLayer(down_layer, down_layer, inhib_tmpl;
                    dist=UniformDist(0.05, hp.inhib_str), density=hp.inhib_den, pre_idx=2, post_idx=2, rng=rng)
    inhib_hidden = SynapseLayer(hidden_layer, hidden_layer, inhib_tmpl;
                    dist=UniformDist(0.05, hp.inhib_str), density=hp.inhib_den, pre_idx=3, post_idx=3, rng=rng)
    inhib_out = SynapseLayer(out_layer, out_layer, inhib_tmpl;
                    dist=UniformDist(0.05, hp.inhib_str), density=hp.inhib_den, pre_idx=4, post_idx=4, rng=rng)

    return LayeredNetwork(
        [up_layer, down_layer, hidden_layer, out_layer],
        [up_to_hidden, down_to_hidden, hidden_to_out,
         inhib_up, inhib_down, inhib_hidden, inhib_out]
    )
end

# ----- Hybrid signal construction -----
function build_hybrid_spiketrain(healthy_rec, infarction_rec; transition_frac=TRANSITION_FRAC, fs=1000.0, Δ_val)
    st_healthy, _, _ = get_spiketrain(healthy_rec.patient, healthy_rec.session; Δ=Δ_val, fs=fs)
    st_infarction, _, _ = get_spiketrain(infarction_rec.patient, infarction_rec.session; Δ=Δ_val, fs=fs)

    total_duration = tsim
    transition_t = total_duration * transition_frac
    transition_samp = round(Int, transition_t * fs)

    healthy_part = filter(s -> s.time < transition_samp, st_healthy)
    infarction_part = [Spike(s.time + transition_samp, s.polarity, s.src_name)
                       for s in st_infarction if s.time + transition_samp < total_duration * fs]

    hybrid = sort(vcat(healthy_part, infarction_part), by=x -> x.time)
    return hybrid, total_duration, transition_t
end

# ----- Simulation with weight tracking -----
function simulate_hybrid(spiketrain, total_duration, hp, rng::AbstractRNG;
                          sample_interval=SAMPLE_INTERVAL, freeze_at=Inf, out_layer_idx=4, fs=1000.0)
    nsteps = Int(round(total_duration / dt))
    max_samp = round(Int, total_duration * fs)

    spiketrain = filter(s -> s.time < max_samp, spiketrain)

    up_pulses = zeros(nsteps)
    down_pulses = zeros(nsteps)

    for spike in spiketrain
        step = Int(floor(spike.time)) + 1
        1 <= step <= nsteps || continue
        spike.polarity ? (up_pulses[step] += hp.pulse_amp) : (down_pulses[step] -= hp.pulse_amp)
    end

    net = build_network(hp, rng)

    input_fn = t -> begin
        idx = clamp(Int(floor(t / dt)) + 1, 1, nsteps)
        n_neur = length(net.neuronlayers)
        v = zeros(n_neur)
        v[1] = up_pulses[idx]
        v[2] = down_pulses[idx]
        v
    end

    sample_steps = Float64[]
    scores = Float64[]
    next_sample = 0.0

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
            next_sample += sample_interval
        end
    end

    runlayers!(net, dt, total_duration; inputfn=input_fn, callback=cb, freeze_at=freeze_at)

    return (times=sample_steps, scores=scores, net=net)
end

# ----- Smoothing -----
smooth(signal, window_size=5) = begin
    half = div(window_size, 2)
    [mean(signal[max(1, i-half):min(end, i+half)]) for i in eachindex(signal)]
end

# ----- Detection -----
function detect_anomaly(signal, times, transition_t; settle=5.0, min_consec_decrease=3, smooth_window=7)
    smoothed = smooth(signal, smooth_window)
    calib_mask = (times .> settle) .& (times .< transition_t)
    calib_idcs = findall(calib_mask)

    alarm_idx = let
        consec_decrease = 0
        found = nothing
        last_calib_idx = calib_idcs[end]
        for i in 2:length(smoothed)
            i <= last_calib_idx && continue
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

    alarm_time = alarm_idx !== nothing ? times[alarm_idx] : -1.0
    alarm_latency = alarm_time > 0 ? alarm_time - transition_t : -1.0

    early_mask = times .< transition_t
    late_mask = times .>= transition_t
    healthy_seg = signal[early_mask]
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
            fp_rate=fp_rate, det_rate=det_rate,
            healthy_mean=mean(healthy_seg), infarction_mean=mean(infarction_seg),
            n_calib=length(calib_idcs))
end

# ----- Single pair evaluation -----
function evaluate_pair(healthy_rec, infarction_rec, hp, rng::AbstractRNG)
    hybrid_train, hybrid_len, transition_t = build_hybrid_spiketrain(
        healthy_rec, infarction_rec; Δ_val=hp.Δ
    )

    result = simulate_hybrid(hybrid_train, hybrid_len, hp, rng)

    det = detect_anomaly(result.scores, result.times, transition_t)

    return merge((transition_t=transition_t,), det)
end

# ----- Evaluation over multiple pairs -----
function evaluate(hp::HyperParams; healthy_pool=healthy_train, infarction_pool=infarction_train,
                  n_pairs=VALIDATION_PAIRS, seed=SEED)
    t_start = time()
    base_seed = param_hash(hp, seed)

    detections = Bool[]
    latencies = Float64[]
    fp_rates = Float64[]
    det_rates = Float64[]
    healthy_means = Float64[]
    infarction_means = Float64[]
    errors = 0

    for pair_idx in 1:n_pairs
        pair_rng = MersenneTwister(base_seed + pair_idx)
        h_rec = sample(pair_rng, healthy_pool)
        i_rec = sample(pair_rng, infarction_pool)
        try
            r = evaluate_pair(h_rec, i_rec, hp, pair_rng)
            push!(detections, r.alarm_latency > 0)
            if r.alarm_latency > 0
                push!(latencies, r.alarm_latency)
            end
            push!(fp_rates, r.fp_rate)
            push!(det_rates, r.det_rate)
            push!(healthy_means, r.healthy_mean)
            push!(infarction_means, r.infarction_mean)
        catch e
            errors += 1
            @warn "Error on pair ($(h_rec.patient),$(i_rec.patient)): $e"
        end
    end

    n_ok = length(detections)
    det_rate_val = n_ok > 0 ? mean(detections) : 0.0
    mean_latency = length(latencies) > 0 ? mean(latencies) : -1.0
    mean_fp = length(fp_rates) > 0 ? mean(fp_rates) : 0.0
    mean_det = length(det_rates) > 0 ? mean(det_rates) : 0.0
    h_mean = length(healthy_means) > 0 ? mean(healthy_means) : 0.0
    i_mean = length(infarction_means) > 0 ? mean(infarction_means) : 0.0

    ε = 1e-8
    separation_raw = abs(i_mean - h_mean) / (abs(h_mean) + ε)
    sep_contrib = tanh(separation_raw)
    latency_penalty = mean_latency > 0 ? 1.0 / (1.0 + mean_latency) : 0.0
    score = det_rate_val * (1.0 - mean_fp) * latency_penalty * (1.0 + sep_contrib)

    return (score=score, det_rate=det_rate_val, mean_latency=mean_latency,
            mean_fp=mean_fp, mean_det=mean_det,
            healthy_mean=h_mean, infarction_mean=i_mean,
            separation=sep_contrib, n_ok=n_ok, n_errors=errors, n_pairs=n_pairs,
            runtime=round(time() - t_start, digits=1))
end

# ----- Evaluate on holdout set -----
function evaluate_holdout(best_hp::HyperParams; n_pairs=VALIDATION_PAIRS)
    println("\n----- Evaluating best on holdout set -----")
    res = evaluate(best_hp; healthy_pool=healthy_holdout, infarction_pool=infarction_holdout,
                   n_pairs=n_pairs, seed=999)
    println(
        "  Holdout det_rate=$(round(res.det_rate, digits=3)) " *
        "latency=$(round(res.mean_latency, digits=2))s " *
        "FP=$(round(res.mean_fp, digits=3)) " *
        "sep=$(round(res.separation, digits=3)) " *
        "score=$(round(res.score, digits=4))"
    )
    return res
end

# ----- Search result type -----
struct SearchResult
    score::Float64
    hp::HyperParams
    det_rate::Float64
    mean_latency::Float64
    mean_fp::Float64
    separation::Float64
end

# ----- Run search over parameter list -----
function runsearch(params, label)
    n = length(params)
    res = Vector{SearchResult}(undef, n)
    best_score = Threads.Atomic{Float64}(0.0)
    p = Progress(n; desc="$label...", showspeed=true)

    Threads.@threads for i in 1:n
        ev = evaluate(params[i])
        sr = SearchResult(ev.score, params[i], ev.det_rate, ev.mean_latency,
                          ev.mean_fp, ev.separation)
        res[i] = sr
        Threads.atomic_max!(best_score, ev.score)
        next!(p; showvalues=[(:best_score, round(best_score[], digits=4)),
                             (:det_rate, round(ev.det_rate, digits=3)),
                             (:latency, round(ev.mean_latency, digits=2))])
    end

    sort!(res, by=r -> r.score, rev=true)
    return res
end

# ----- Print results -----
function print_results(results, n=5)
    println("\n----- Top $n parameter sets -----")
    for (i, r) in enumerate(results[1:min(n, length(results))])
        hp = r.hp
        println("\nRank $i --- score=$(round(r.score, digits=4))  det=$(round(r.det_rate, digits=3))  lat=$(round(r.mean_latency, digits=2))s  FP=$(round(r.mean_fp, digits=3))  sep=$(round(r.separation, digits=3))")
        println("   Δ=$(round(hp.Δ, digits=4))  pulse_amp=$(round(hp.pulse_amp, digits=1))")
        println("   ltp_rate=$(round(hp.ltp_rate, digits=4))  ltp_rate_out=$(round(hp.ltp_rate_out, digits=4))")
        println("   τ_s=$(round(hp.τ_s, digits=2))  τ_s_output=$(round(hp.τ_s_output, digits=2))")
        println("   τ_pretrace=$(round(hp.τ_pretrace, digits=2))  τ_posttrace=$(round(hp.τ_posttrace, digits=2))")
        println("   R_m_input=$(round(hp.R_m_input, digits=3))  R_m_output=$(round(hp.R_m_output, digits=3))")
        println("   τ_m_input=$(round(hp.τ_m_input, digits=3))  τ_m_output=$(round(hp.τ_m_output, digits=3))")
        println("   τ_ref_input=$(round(hp.τ_ref_input, digits=3))  τ_ref_output=$(round(hp.τ_ref_output, digits=3))")
        println("   N=$(hp.N)  density=$(round(hp.density, digits=3))")
        println("   inhib_den=$(round(hp.inhib_den, digits=3))  inhib_str=$(round(hp.inhib_str, digits=3))")
    end
end

# ----- Random Search -----
println(
    "\n  Search: $(SEARCH_ITERATIONS) random + $(FOCUSED_TOPK)x$(FOCUSED_ITERATIONS) focused"
)
println(
    "  Per-eval: $(VALIDATION_PAIRS) pairs on training set ($(length(healthy_train))H/$(length(infarction_train))I)"
)
println(
    "  Holdout: $(length(healthy_holdout))H/$(length(infarction_holdout))I for final validation"
)

rng = MersenneTwister(67)
random_params = [sample_params_random(rng) for _ in 1:SEARCH_ITERATIONS]
random_results = runsearch(random_params, "1. Random Search")
print_results(random_results)

# ----- Focused Search -----
focused_rng = MersenneTwister(69)
focused_params = HyperParams[]

for i in 1:FOCUSED_TOPK
    hp = random_results[i].hp
    append!(
        focused_params,
        [sample_params_focused(focused_rng, hp; frac=0.15) for _ in 1:FOCUSED_ITERATIONS]
    )
end
focused_results = runsearch(focused_params, "2. Focused Search")

all_results = sort!(vcat(random_results, focused_results), by=r -> r.score, rev=true)
println("\n----- Final top 10 -----")
print_results(all_results, 10)

# ----- Holdout evaluation -----
if length(all_results) > 0
    best = all_results[1].hp
    holdout_res = evaluate_holdout(best)

    println("\n----- Best params -----")
    println("const Δ = $(best.Δ)")
    println("const ltp_rate = $(best.ltp_rate)")
    println("const ltp_rate_out = $(best.ltp_rate_out)")
    println("const τ_s = $(best.τ_s)")
    println("const τ_m_input = $(best.τ_m_input)")
    println("const τ_ref_output = $(best.τ_ref_output)")
    println("const inhib_str = $(best.inhib_str)")
    println("const pulse_amp = $(best.pulse_amp)")
    println("const τ_pretrace = $(best.τ_pretrace)")
    println("const R_m_input = $(best.R_m_input)")
    println("const R_m_output = $(best.R_m_output)")
    println("const τ_m_output = $(best.τ_m_output)")
    println("const τ_ref_input = $(best.τ_ref_input)")
    println("const inhib_den = $(best.inhib_den)")
    println("const τ_s_output = $(best.τ_s_output)")
    println("const N = $(best.N)")
    println("const density = $(best.density)")
    println("const τ_posttrace = $(best.τ_posttrace)")
end
