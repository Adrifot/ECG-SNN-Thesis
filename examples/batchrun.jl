include("../modules/Layers.jl")
include("../modules/Signals.jl")
include("../modules/Registry.jl")

using .Signals
using .Layers
using .Layers.Neurons
using .Layers.Synapses
using .Layers.Utils

using Statistics
using Random
using ProgressMeter

using .Registry

db_root = "./ecg-db"
all_records = build_registry(db_root)

labelled = filter(r -> r.label != :unknown, all_records)

println("Found $(length(all_records)) total sessions")
println("   healthy: $(count(r -> r.label == :healthy, labelled))")
println("   infarction: $(count(r -> r.label == :infarction, labelled))")
println("   unknown: $(count(r -> r.label == :unknown, all_records))")

# === Fixed Parameter Set ===
const Δ       = 0.075
const fs      = 1000.0
const N       = 25
const dt      = 0.001
const tsim    = 25.0
const SEED    = 42
const PULSE_AMP = 50.0

function build_network(; seed=SEED)
    Random.seed!(seed)

    up_template = Neuron("up_input"; R_m=0.5, τ_m=2.0, τ_s=60.0, τ_ref=4.0,
                    τ_pretrace=30.0, τ_posttrace=10.0)
    down_template = Neuron("down_input"; R_m=0.5, τ_m=2.0, τ_s=60.0, τ_ref=4.0,
                τ_pretrace=30.0, τ_posttrace=10.0, isreverse=true)
    out_template = Neuron("output"; R_m=3.0, τ_m=6.0, τ_s=6.0, τ_ref=2.0,
                τ_pretrace=30.0, τ_posttrace=10.0)

    exc_syn_template   = Synapse(1, 2; learningrate=0.15, wmax=1.0)
    inhib_template = Synapse(1, 1; learningrate=0.0,  wmax=1.0, isinhibitory=true)

    up_layer = NeuronLayer(N, up_template; name="up_input", V_thresh_dev=0.1, R_m_dev=0.1, τ_m_dev=0.15)
    down_layer = NeuronLayer(N, down_template; name="down_input", V_thresh_dev=0.1, R_m_dev=0.1, τ_m_dev=0.15)
    out_layer = NeuronLayer(N, out_template;  name="output", V_thresh_dev=0.1, R_m_dev=0.1, τ_m_dev=0.15)

    up_to_out = SynapseLayer(up_layer, out_layer, exc_syn_template;
        dist=NormalDist(0.5, 0.2), density=0.75, pre_idx=1, post_idx=3)

    down_to_out = SynapseLayer(down_layer, out_layer, exc_syn_template;
        dist=NormalDist(0.5, 0.2), density=0.75, pre_idx=2, post_idx=3)

    inhib_up = SynapseLayer(up_layer,   up_layer,   inhib_template;
        dist=UniformDist(0.1, 0.2), density=0.25, pre_idx=1, post_idx=1)
    inhib_down = SynapseLayer(down_layer, down_layer, inhib_template;
        dist=UniformDist(0.1, 0.2), density=0.25, pre_idx=2, post_idx=2)
    inhib_out = SynapseLayer(out_layer,  out_layer,  inhib_template;
        dist=UniformDist(0.1, 0.2), density=0.25, pre_idx=3, post_idx=3)

    return LayeredNetwork(
        [up_layer, down_layer, out_layer],
        [up_to_out, down_to_out, inhib_up, inhib_down, inhib_out]
    )
end

function run_patient(patient, session, label)
    spiketrain, siglen, _ = get_spiketrain(patient, session; Δ=Δ, fs=fs)
    tsim_actual = min(tsim, siglen / fs)      
    nsteps = Int(round(tsim_actual / dt))     

    up_pulses = zeros(nsteps)
    down_pulses = zeros(nsteps)
    for spike in spiketrain
        step = Int(floor(spike.time)) + 1
        1 <= step <= nsteps || continue
        if spike.polarity
            up_pulses[step]   += PULSE_AMP
        else
            down_pulses[step] -= PULSE_AMP
        end
    end

    input_fn = t -> begin
        idx = clamp(Int(floor(t / dt)) + 1, 1, nsteps)
        return [up_pulses[idx], down_pulses[idx], 0.0]
    end

    net = build_network(seed=hash((patient, session)))
    runlayers!(net, dt, tsim; inputfn=input_fn)

    w_up = net.synapselayers[1].ws
    w_down = net.synapselayers[2].ws
    up_count = count(>(0), up_pulses)
    down_count = count(<(0), down_pulses)

    return (
        patient = patient,
        session = session,
        label = label,
        up_spikes = up_count,
        down_spikes = down_count,
        total_spikes = up_count + down_count,
        spike_ratio = up_count / max(down_count, 1),
        mean_w_up = mean(w_up),
        mean_w_down = mean(w_down),
        pathway_bias = mean(w_up) - mean(w_down),
        dead_frac_up = count(iszero, w_up)  / length(w_up),
        dead_frac_down = count(iszero, w_down) / length(w_down),
        w_entropy_up = let w = w_up[w_up .> 0] -sum(w .* log.(w)) / length(w_up) end,
        w_entropy_down= let w = w_down[w_down .> 0] -sum(w .* log.(w)) / length(w_down) end
    )
end

# === Main Loop ===
results = []
p = Progress(length(labelled); desc="Running patients...", showspeed=true)
for r in labelled
    result = run_patient(r.patient, r.session, r.label)
    push!(results, result)
    next!(p; showvalues = [(:patient, r.patient), (:label, r.label), 
                           (:spikes, result.total_spikes)])
end

# === Summary ===
println("\n", "="^85)
println(rpad("patient", 10), rpad("label", 12), rpad("spikes", 8),
        rpad("ratio", 8), rpad("mean_w", 9), rpad("bias", 9),
        rpad("entropy_up", 12), "dead_frac")
println("-"^85)
for r in results
    mean_w = round((r.mean_w_up + r.mean_w_down) / 2, digits=4)
    println(
        rpad(r.patient, 10),
        rpad(string(r.label), 12),
        rpad(r.total_spikes, 8),
        rpad(round(r.spike_ratio, digits=3), 8),
        rpad(mean_w, 9),
        rpad(round(r.pathway_bias, digits=4), 9),
        rpad(round(r.w_entropy_up, digits=4), 12),
        round(r.dead_frac_up, digits=4)
    )
end
println("="^85)

for lbl in [:healthy, :infarction]
    group = filter(r -> r.label == lbl, results)
    mean_spikes = mean([r.total_spikes for r in group])
    mean_w = mean([(r.mean_w_up + r.mean_w_down)/2 for r in group])
    mean_ratio = mean([r.spike_ratio for r in group])
    println("\n$(lbl): n=$(length(group)), mean spikes=$(round(mean_spikes, digits=1)), " *
            "mean weight=$(round(mean_w, digits=4)), mean ratio=$(round(mean_ratio, digits=3))")
end

healthy_w = [(r.mean_w_up + r.mean_w_down)/2 for r in results if r.label == :healthy   && r.total_spikes > 0]
infarction_w = [(r.mean_w_up + r.mean_w_down)/2 for r in results if r.label == :infarction && r.total_spikes > 0]
healthy_s = [r.total_spikes for r in results if r.label == :healthy    && r.total_spikes > 0]
infarction_s = [r.total_spikes for r in results if r.label == :infarction && r.total_spikes > 0]

function cohens_d(a, b)
    pooled_std = sqrt((std(a)^2 + std(b)^2) / 2)
    return (mean(a) - mean(b)) / pooled_std
end

function overlap_coefficient(a, b; bins=50)
    lo = min(minimum(a), minimum(b))
    hi = max(maximum(a), maximum(b))
    step = (hi - lo) / bins
    ha = [count(x -> lo + (i-1)*step <= x < lo + i*step, a) / length(a) for i in 1:bins]
    hb = [count(x -> lo + (i-1)*step <= x < lo + i*step, b) / length(b) for i in 1:bins]
    return sum(min.(ha, hb))
end

println("\n=== Separation Analysis ===")
println("Mean weight — healthy: $(round(mean(healthy_w), digits=4)), " *
        "infarction: $(round(mean(infarction_w), digits=4))")
println("Std  weight — healthy: $(round(std(healthy_w),  digits=4)), " *
        "infarction: $(round(std(infarction_w),  digits=4))")
println("Cohen's d (weight): $(round(cohens_d(infarction_w, healthy_w), digits=3))")
println("Cohen's d (spikes): $(round(cohens_d(infarction_s, healthy_s), digits=3))")
println("Overlap coefficient (weight): $(round(overlap_coefficient(healthy_w, infarction_w), digits=3))")
println("Overlap coefficient (spikes): $(round(overlap_coefficient(healthy_s,  infarction_s),  digits=3))")

