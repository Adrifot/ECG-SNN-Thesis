using DSP

PATIENT = "104"
SESSION = "s0306lre"

dt = 1.0

struct Spike 
    time::Float64
    polarity::Bool
end

struct OutputSpike
    time::Float64
    neuron_name::String
end

function delta_modulation(signal; Δ=100)
    n = length(signal)
    last_spike_lvl = signal[1]
    spiketrain = Spike[]
    for t in 2:n
        diff = signal[t] - last_spike_lvl 
        if diff >= Δ
            push!(spiketrain, Spike(t, true))
            last_spike_lvl += Δ 
        elseif diff <= -Δ
            push!(spiketrain, Spike(t, false))
            last_spike_lvl -= Δ
        end
    end
    return spiketrain
end


function load_raw_signal(patient, session)
    path = "./ecg-db/patient$(patient)/$(session).dat"
    if !isfile(path)
        error("File not found: $(path)")
    end
    raw_data = reinterpret(Int16, read(path))
    n_channels = 16
    data_matrix = reshape(raw_data, n_channels, :)
    return Float64.(data_matrix[2, :])
end


function get_filtered_signal(signal; lowcut=0.01, highcut=40, fs=1000)
    pass = Bandpass(lowcut, highcut)
    method = Butterworth(4)
    return filtfilt(digitalfilter(pass, method; fs=fs), signal)
end

mutable struct Neuron
    name::String # Neuron identifier
    τ_m::Float64 # Membrane time constant
    τ_ref::Float64 # Refractory period
    V_rest::Float64 # Resting potential
    V_thresh::Float64 # Threshold potential
    V_reset::Float64 # Reset potential
    R_m::Float64 # Resistance
    v::Float64 # Current membrane potential
    t_ref::Float64 # Current remaining refractory time
    i_ext::Float64 # Current current
    τ_s::Float64 # Synaptic decay constant
    w::Float64 # Synaptic weight
    is_reverse::Bool # If it's sensible to the reverse polarity
end

function Neuron(; name="neuron", τ_m=20.0, τ_ref=2.0, V_rest=-70.0, 
                V_thresh=-50.0, V_reset=-70.0, R_m=1.0, 
                τ_s=5.0, w=15.0, is_reverse=false)
    return Neuron(name, τ_m, τ_ref, V_rest, V_thresh, V_reset, 
                R_m, 0.0, 0.0, 0.0, τ_s, w, is_reverse)
end

function update!(n::Neuron, spike_type::Int, dt::Float64)
    # 1. Update synaptic current: dI/dt = -I/τ_s
    n.i_ext += (-n.i_ext / n.τ_s) * dt

    # 2. Inject new current (Scale this down or increase τ_s decay)
    if spike_type != 0
        mult = n.is_reverse ? -1 : 1
        # Only inject if the polarity matches the detector's
        if (spike_type * mult) > 0
            n.i_ext += n.w 
        end
    end

    # 3. Refractory handling
    if n.t_ref > 0
        n.v = n.V_reset
        n.t_ref -= dt
        return false
    end

    # 4. LIF Equation
    dv = (-(n.v - n.V_rest) + n.R_m * n.i_ext) / n.τ_m * dt
    n.v += dv

    # 5. Threshold check
    if n.v ≥ n.V_thresh
        n.v = n.V_reset
        n.t_ref = n.τ_ref
        n.i_ext = 0 
        return true
    end

    return false
end

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

function get_spiketrain(PATIENT, SESSION; Δ=100)
    raw_sig = load_raw_signal(PATIENT, SESSION)
    filt_sig = get_filtered_signal(raw_sig)
    spiketrain = delta_modulation(filt_sig; Δ=Δ)
    return spiketrain, length(filt_sig), filt_sig
end

function run(N, spiketrain, neurons, dt)
    results = [Float64[] for _ in neurons]
    output_spikes = OutputSpike[]
    
    for t ∈ 1:N 
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



using Plots
plotly()

# Get spiketrain and signal length
spiketrain, N, filt_sig = get_spiketrain(PATIENT, SESSION; Δ=100)

# Run simulation with multiple neurons
neurons = [QRS_up, QRS_down]
results, output_spikes = run(N, spiketrain, neurons, dt)
results_up, results_down = results[1], results[2]

# println("Total output spikes: ", length(output_spikes))
# println("First 10 output spikes:")
# for spike in output_spikes[1:min(10, length(output_spikes))]
#     println("  t=$(spike.time), neuron=$(spike.neuron_name)")
# end

# --- Visualization Parameters ---
fs = 1000          # Sampling frequency
duration = 2.0     # Seconds to display

# Calculate middle seconds
total_samples = length(filt_sig)
samples_to_plot = Int(duration * fs)
middle_idx = total_samples ÷ 2
start_idx = max(1, middle_idx - samples_to_plot ÷ 2)
end_idx = min(total_samples, start_idx + samples_to_plot - 1)
time_axis = (start_idx-1:end_idx-1) ./ fs

# --- Subplot 1: Filtered ECG ---
p1 = plot(time_axis, filt_sig[start_idx:end_idx], 
          linecolor=:blue, ylabel="mV", title="Filtered ECG (Lead II)", 
          legend=false, grid=true)

# --- Subplot 2: Up-Detector Neuron ---
p2 = plot(time_axis, results_up[start_idx:end_idx], 
          linecolor=:green, ylabel="V_m (Up)", label="Up Neuron")
hline!(p2, [QRS_up.V_thresh], line=:dash, color=:red, label="Threshold")

# --- Subplot 3: Down-Detector Neuron ---
p3 = plot(time_axis, results_down[start_idx:end_idx], 
          linecolor=:red, ylabel="V_m (Down)", label="Down Neuron")
hline!(p3, [QRS_down.V_thresh], line=:dash, color=:red, label="Threshold")

final_plot = plot(p1, p2, p3, layout=(3, 1), size=(900, 700), 
                  xlabel="Time (seconds)", link=:x)

display(final_plot)