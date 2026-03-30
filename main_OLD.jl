# !!! FILE DEPRECATED !!!

include("modules/Neurons.jl")
include("modules/Signals.jl")

using .Neurons
using .Signals

using Revise

PATIENT = "104"
SESSION = "s0306lre"

dt = 1.0

QRS_up = Neuron(;
    name="QRS_up",
    τ_m=100.0,   
    τ_ref=200.0,   
    V_rest=0.0, 
    V_thresh=40.0, 
    V_reset=0.0, 
    τ_s=30.0,      
    w=15.0,
    isreverse=false
)

QRS_down = Neuron(;
    name="QRS_down",
    τ_m=100.0,      
    τ_ref=200.0,   
    V_rest=0.0, 
    V_thresh=40.0, 
    V_reset=0.0, 
    τ_s=30.0,      
    w=15.0,
    isreverse=true
)

function run(T_end, spiketrain, neurons::Vector{T}, dt) where T <: Neuron
    n_steps = Int(floor(T_end / dt))
    n_neurons = length(neurons)
    
    # 1. Pre-allocate results matrix (Neurons are rows, Time is columns)
    results = Matrix{Float64}(undef, n_neurons, n_steps)
    
    # 2. Pre-allocate output spike buffer
    output_spikes = Vector{OutputSpike}()
    sizehint!(output_spikes, 1000) 

    # 3. Sort spikes by time 
    spikes = sort(spiketrain, by = s -> s.time)
    
    spike_idx = 1
    t = 0.0

    # 4. Main Simulation Loop
    for step in 1:n_steps
        # Find if there's a spike in this time window
        current_spike = nothing
        if spike_idx <= length(spikes) && spikes[spike_idx].time <= t
            current_spike = spikes[spike_idx]
            spike_idx += 1
        end

        # 5. Inner Neuron Loop
        for j in 1:n_neurons
            neuron = neurons[j]
            fired = update!(neuron, current_spike, dt)
            results[j, step] = neuron.v

            if fired
                push!(output_spikes, OutputSpike(t, neuron.name))
            end
        end
        
        t += dt
    end

    return results, output_spikes
end

using Plots
plotly()   

function run_pipeline(PATIENT, SESSION, neurons; Δ=100, dt=1.0)
    spiketrain, N, filt_sig = get_spiketrain(PATIENT, SESSION; Δ=Δ)
    results, output_spikes = run(N, spiketrain, neurons, dt)

    return (
        filt_sig = filt_sig,
        results_up = results[1],
        results_down = results[2]
    )
end

function extract_window(filt_sig, results_up, results_down; fs=1000, duration=2.0)
    total_samples = length(filt_sig)
    samples_to_plot = Int(duration * fs)

    middle_idx = total_samples ÷ 2
    start_idx = max(1, middle_idx - samples_to_plot ÷ 2)
    end_idx = min(total_samples, start_idx + samples_to_plot - 1)

    time_axis = (start_idx-1:end_idx-1) ./ fs

    return (
        time_axis = time_axis,
        filt_sig = filt_sig[start_idx:end_idx],
        results_up = results_up[start_idx:end_idx],
        results_down = results_down[start_idx:end_idx]
    )
end

function make_plot(data, QRS_up, QRS_down)
    p1 = plot(data.time_axis, data.filt_sig,
        linecolor=:blue, ylabel="mV",
        title="Filtered ECG (Lead II)",
        legend=false, grid=true)

    p2 = plot(data.time_axis, data.results_up,
        linecolor=:green, ylabel="V_m (Up)",
        label="Up Neuron")
    hline!(p2, [QRS_up.V_thresh],
        line=:dash, color=:red, label="Threshold")

    p3 = plot(data.time_axis, data.results_down,
        linecolor=:red, ylabel="V_m (Down)",
        label="Down Neuron")
    hline!(p3, [QRS_down.V_thresh],
        line=:dash, color=:red, label="Threshold")

    plot(p1, p2, p3,
        layout=(3,1),
        size=(900,700),
        xlabel="Time (seconds)",
        link=:x)
end

neurons = [QRS_up, QRS_down]

pipeline_data = run_pipeline(PATIENT, SESSION, neurons; Δ=100, dt=dt)

window = extract_window(
    pipeline_data.filt_sig,
    pipeline_data.results_up,
    pipeline_data.results_down;
    fs=1000,
    duration=2.0
)

make_plot(window, QRS_up, QRS_down) |> display