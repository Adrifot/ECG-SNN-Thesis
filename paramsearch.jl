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