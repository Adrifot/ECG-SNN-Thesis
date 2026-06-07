include("./modules/Layers.jl")
include("./modules/Signals.jl")
include("./modules/Registry.jl")

using .Signals
using .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils
using .Registry

using Random
using ProgressMeter
using Statistics, LinearAlgebra

# ----- Getting all the patients -----
db_root = "./ecg-db"
all_records = build_registry(db_root)
labelled = filter(r -> r.label != :unknown, all_records)

Random.seed!(42)

healthy_records = filter(r -> r.label == :healthy, labelled)
infarction_records = filter(r -> r.label == :infarction, labelled)

subset = vcat(
    sample(healthy_records, min(80, length(healthy_records)); replace=false),
    sample(infarction_records, min(240, length(infarction_records)); replace=false)
)

println(
    "Searching on $(length(subset)) patients " *
    "$(count(r->r.label==:healthy, subset)) healthy, " *    
    "$(count(r->r.label==:infarction, subset)) infarction"
)

# ----- HyperParams -----
struct HyperParams
    Δ::Float64
    pulse_amp::Float64
    learningrate::Float64
    τ_s_input::Float64
    τ_pretrace::Float64
    τ_posttrace::Float64
    R_m_input::Float64
    R_m_output::Float64
    τ_m_input::Float64
    τ_m_output::Float64
    τ_s_output::Float64
    τ_ref_input::Float64
    τ_ref_output::Float64
    N::Int
    density::Float64
    inhib_density::Float64
    inhib_strnegth::Float64
end

# ----- Full Random Search -----
function getrnd(a, b, rng)
    return rand(rng) * (b - a) + a
end

function sample_params_random(rng)
    HyperParams(
        getrnd(0.01, 0.25, rng), # Δ
        getrnd(10.0, 300.0, rng), # pulse_amp
        getrnd(0.05, 0.25, rng), # learningrate
        getrnd(1.0, 50.0, rng), # τ_s_input
        getrnd(10.0, 100.0, rng), # τ_pretrace
        getrnd(10.0, 100.0, rng), # τ_posttrace
        getrnd(0.1, 10.0, rng), # R_m_input
        getrnd(0.1, 10.0, rng), # R_m_output
        getrnd(0.5, 10.0, rng), # τ_m_input
        getrnd(0.1, 10.0, rng), # τ_m_output
        getrnd(1.0, 50.0, rng), # τ_s_output
        getrnd(0.5, 10.0, rng), # τ_ref_input
        getrnd(0.5, 10.0, rng), # τ_ref_output
        rand(rng, [10, 15, 20, 25, 30, 40, 50]), # N
        getrnd(0.5, 0.99, rng), # density
        getrnd(0.1, 0.9, rng), #inhib_density
        getrnd(0.1, 0.9, rng) #inhib_strength
    )
end

# ----- Focused search -----
function sample_params_focused(rng, best::HyperParams; frac=0.15)
    function perturb(val, lo, hi)
        clamp(
            val * (1 + (2*rand(rng) - 1) * frac),
            lo, hi
        )
    end
    HyperParams(
        perturb(best.Δ, 0.01, 0.25),
        perturb(best.pulse_amp, 10.0, 300.0),
        perturb(best.learningrate, 0.05, 0.25),
        perturb(best.τ_s_input, 1.0, 50.0),
        perturb(best.τ_pretrace, 10.0, 100.0),
        perturb(best.τ_posttrace, 10.0, 100.0),
        perturb(best.R_m_input, 0.1, 10.0),
        perturb(best.R_m_output, 0.1, 10.0),
        perturb(best.τ_m_input, 0.5, 10.0),
        perturb(best.τ_m_output, 0.1, 10.0),
        perturb(best.τ_s_output, 1.0, 50.0),
        perturb(best.τ_ref_input, 0.5, 10.0),
        perturb(best.τ_ref_output, 0.5, 10.0),
        best.N, 
        perturb(best.density, 0.5, 0.99),
        perturb(best.inhib_density, 0.1, 0.9),
        perturb(best.inhib_strength, 0.1, 0.9)
    )
end

# ----- Network -----
const dt = 0.001
const tsim = 25.0
const SEED = 42

function build_network(hp::HyperParams)
    Random.seed!(SEED)

    # Neuron templates
    up_template = Neuron("up_in"; R_m=hp.R_m_input, τ_m=hp.τ_m_input, τ_s=hp.τ_s_input, 
                    τ_ref=hp.τ_ref_input, τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace)
    down_template = Neuron("down_in"; R_m=hp.R_m_input, τ_m=hp.τ_m_input, τ_s=hp.τ_s_input, 
                    τ_ref=hp.τ_ref_input, τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace,
                    isreverse=true)
    out_template = Neuron("out"; R_m=hp.R_m_output, τ_m=hp.τ_m_output, τ_s=hp.τ_s_output,
                    τ_ref=hp.τ_ref_output, τ_pretrace=hp.τ_pretrace, τ_posttrace=hp.τ_posttrace)

    # Synapse templates
    syn_template = Synapse(1, 2; learningrate=hp.learningrate, wmax=1.0)
    inhib_template = Synapse(1, 1; learningrate=0.0, wmax=1.0, isinhibitory=true)

    # Neuron layers
    uplayer = NeuronLayer(hp.N, up_template; name="up_in", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
    downlayer = NeuronLayer(hp.N, down_template; name="down_in", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
    outlayer = NeuronLayer(hp.N, out_template; name="out", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)

    # Synaptic layers
    uptout = SynapseLayer(uplayer, outlayer, syn_template; 
                dist=NormalDist(0.5, 0.2), density=hp.density,
                pre_idx=1, post_idx=3)
    downtout = SynapseLayer(uplayer, outlayer, syn_template; 
                dist=NormalDist(0.5, 0.2), density=hp.density,
                pre_idx=1, post_idx=3)
    inhibup = SynapseLayer(uplayer, uplayer, inhib_template;
                dist=UniformDist(0.05, hp.inhib_strength), density=hp.inhib_density, pre_idx=1, post_idx=1)
    inhibdown = SynapseLayer(downlayer, downlayer, inhib_template;
                dist=UniformDist(0.05, hp.inhib_strength), density=hp.inhib_density, pre_idx=2, post_idx=2)
    inhibout = SynapseLayer(outlayer, outlayer, inhib_template;
                dist=UniformDist(0.05, hp.inhib_strength), density=hp.inhib_density, pre_idx=3, post_idx=3)

    return LayeredNetwork(
        [uplayer, downlayer, outlayer],
        [uptout, downtout, inhibup, inhibdown, inhibout]
    )
end 

function runone(rec, hp::HyperParams)
    spiketrain, siglen, _ = get_spiketrain(rec.patient, rec.session, Δ=hp.Δ, fs=1000.0, gap=100.0)
    tsim_actual = min(tsim, siglen / 1000.0)
    nstepts = Int(round(tsim_actual / dt))

    uppulses = zeros(nsteps)
    downpulses = zeros(nsteps)

    for spike in spiketrain
        step = Int(floor(spike.time))
        1 ≤ step ≤ nsteps || continue
        spike.polarity ? (uppulses[step] += hp.pulse_amp) : (downpulses[step] -= hp.pulse_amp)
    end

    input_fn = t -> begin
        idx = clamp(Int(floor(t/dt)) + 1, 1, nsteps)
        [uppulses[idx], downpulses[idx], 0.0]
    end

    net = build_network(hp)
    runlayers!(net, dt, tsim_actual; inputfn=input_fn)

    w_up = net.synapselayers[1].ws
    w_down = net.synapselayers[2].ws
    return (mean(w_up) + mean(w_down)) / 2
end

function cohens_d(a, b)
    pooled = sqrt((std(a)^2 + std(b)^2) / 2)
    pooled < 1e-10 && return 0.0
    return abs(mean(a) - mean(b)) / pooled
end

function evaluate(hp::HyperParams)
    t_start = time()
    weights = Dict(:healthy => Float64[], :infarction => Float64[])
    errors = Int[0, 0] #[healthy, infarction]

    for rec in subset
        try
            w = wunone(rec, hp)
            push!(weights[rec.label], 2)
        catch e
            if rec.label == :healthy 
                errors[1] += 1
            else 
                errors[2] += 1
            end
            @warn "Error on $(rec.patient): $e"
        end
    end

    n_h = length(weights[:healthy])
    n_i = length(weights[:infarction])
    d = (n_h<5 || n_i < 5) ? 0.0 : cohens_d(weights[:healthy], weights[:infarction])

    return (d=d, n_healthy=n_h, n_infarction=n_i, errors_h=errors[1], errors_i=errors[2],
            runtime=round(time() - t_start, digits=1))
end



