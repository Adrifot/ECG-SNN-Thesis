include("modules/Neurons.jl")
include("modules/Signals.jl")

using .Neurons
using .Signals

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
    is_reverse=false
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
    is_reverse=true
)

function run(N, spiketrain, neurons, dt)
    results = [Float64[] for _ in neurons]
    output_spikes = OutputSpike[]
    
    for t in 1:N 
        s_at_t = filter(s -> s.time == t, spiketrain)
        pol = isempty(s_at_t) ? 0 : (s_at_t[1].polarity ? 1 : -1)
        
        for (i, neuron) in enumerate(neurons)
            fired = update!(neuron, pol, dt)
            push!(results[i], neuron.v)
            
            if fired
                push!(output_spikes, OutputSpike(t, neuron.name))
            end
        end
    end
    
    return results, output_spikes
end

# --- PLOTTING ---
using Plots
plotly()

spiketrain, N, filt_sig = get_spiketrain(PATIENT, SESSION; Δ=100)

neurons = [QRS_up, QRS_down]
results, output_spikes = run(N, spiketrain, neurons, dt)
results_up, results_down = results[1], results[2]

# Visualization Parameters 
fs = 1000          
duration = 2.0 # seconds

# Calculate middle seconds
total_samples = length(filt_sig)
samples_to_plot = Int(duration * fs)
middle_idx = total_samples ÷ 2
start_idx = max(1, middle_idx - samples_to_plot ÷ 2)
end_idx = min(total_samples, start_idx + samples_to_plot - 1)
time_axis = (start_idx-1:end_idx-1) ./ fs

# Subplot 1: Filtered ECG 
p1 = plot(time_axis, filt_sig[start_idx:end_idx], 
          linecolor=:blue, ylabel="mV", title="Filtered ECG (Lead II)", 
          legend=false, grid=true)

# Subplot 2: Up-Detector Neuron 
p2 = plot(time_axis, results_up[start_idx:end_idx], 
          linecolor=:green, ylabel="V_m (Up)", label="Up Neuron")
hline!(p2, [QRS_up.V_thresh], line=:dash, color=:red, label="Threshold")

# Subplot 3: Down-Detector Neuron
p3 = plot(time_axis, results_down[start_idx:end_idx], 
          linecolor=:red, ylabel="V_m (Down)", label="Down Neuron")
hline!(p3, [QRS_down.V_thresh], line=:dash, color=:red, label="Threshold")

final_plot = plot(p1, p2, p3, layout=(3, 1), size=(900, 700), 
                  xlabel="Time (seconds)", link=:x)

display(final_plot)