using DSP

PATIENT = "169"
SESSION = "s0329lre"

struct Spike 
    time::Float64
    polarity::Bool
end

function delta_modulation(signal; Δ=120)
    n = length(signal)
    last_spike_lvl = signal[1]
    spiketrain = Spike[]

    for t in 2:n
        diff = signal[t] - last_spike_lvl
        if diff ≥ Δ
            spike = Spike(t, true)
            push!(spiketrain, spike)
            last_spike_lvl = signal[t]
        elseif diff ≤ -Δ
            spike = Spike(t, false)
            push!(spiketrain, spike)
            last_spike_lvl = signal[t]
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


function get_filtered_signal(signal; lowcut=0.5, highcut=40, fs=1000)
    pass = Bandpass(lowcut, highcut)
    method = Butterworth(4)
    return filtfilt(digitalfilter(pass, method; fs=fs), signal)
end

mutable struct Neuron
    τ_m::Float64 # Time constant
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
end

function Neuron(; τ_m=20.0, τ_ref=2.0, V_rest=-70.0, 
                V_thresh=-50.0, V_reset=-70.0, R_m=1.0, 
                τ_s=5.0, w=15.0)
    return Neuron(τ_m, τ_ref, V_rest, V_thresh, V_reset, 
                R_m, V_rest, 0.0, 0.0, τ_s, w)
end

function update!(n::Neuron, spike_type::Int, dt::Float64; is_reverse=false)
    # Update synaptic current: dI/dt = -I/τ_s
    n.i_ext += (-n.i_ext / n.τ_s) * dt

    # Inject new current when spike occurs
    if spike_type != 0
        mult = is_reverse ? -1 : 1
        excitation = (spike_type * mult) > 0
        force = excitation ? n.w : n.w/1.25
        n.i_ext += (spike_type * mult) * force
    end

    # If in refractory period, voltage stuck at V_ref
    if n.t_ref > 0
        n.v = n.V_reset
        n.t_ref -= dt
        return false
    end

    # LIF Equation (leakage towards rest)
    dv = (-(n.v - n.V_rest) + n.R_m * n.i_ext) / n.τ_m * dt
    n.v += dv

    n.v = max(n.v, -10.0)

    # Threshold check
    if n.v ≥ n.V_thresh
        n.v = n.V_reset
        n.t_ref = n.τ_ref
        return true
    end
    return false
end

dt = 0.1

up_detector = Neuron(;τ_m=30.0, τ_ref=10.0, V_rest=0.0, V_thresh=7.5, V_reset=0.0, τ_s=20.0, w=35.0, R_m=1.0)
down_detector = Neuron(;τ_m=30.0, τ_ref=10.0, V_rest=0.0, V_thresh=7.5, V_reset=0.0, τ_s=20.0, w=35.0, R_m=1.0)

results_up = Float64[]
results_down = Float64[]

raw_sig = load_raw_signal(PATIENT, SESSION)
filt_sig = get_filtered_signal(raw_sig)
spiketrain = delta_modulation(filt_sig; Δ=125)

N = length(filt_sig)

for t ∈ 1:N 
    s_at_t = filter(s -> s.time == t, spiketrain)
    pol = isempty(s_at_t) ? 0 : (s_at_t[1].polarity ? 1 : -1)
    fired_up = update!(up_detector, pol, dt; is_reverse=false)
    fired_down = update!(down_detector, pol, dt; is_reverse=true)
    push!(results_up, up_detector.v)
    push!(results_down, down_detector.v)
end

using Plots
plotly()

# --- Visualization Parameters ---
fs = 1000          # Sampling frequency
duration = 4.0     # Seconds to display
dt = 1.0           # Step size

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
hline!(p2, [up_detector.V_thresh], line=:dash, color=:red, label="Threshold")

# --- Subplot 3: Down-Detector Neuron ---
p3 = plot(time_axis, results_down[start_idx:end_idx], 
          linecolor=:red, ylabel="V_m (Down)", label="Down Neuron")
hline!(p3, [down_detector.V_thresh], line=:dash, color=:red, label="Threshold")

final_plot = plot(p1, p2, p3, layout=(3, 1), size=(900, 700), 
                  xlabel="Time (seconds)", link=:x)

display(final_plot)