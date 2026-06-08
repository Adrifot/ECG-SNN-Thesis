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
const τ_ref_outt = 1.32
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










