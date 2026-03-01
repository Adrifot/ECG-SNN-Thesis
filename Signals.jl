"""
Signals.jl

Utilities for loading and preprocessing ECG recordings,
including bandpass filtering and delta modulation spike encoding.
"""

module Signals

export load_raw_signal, get_filtered_signal, get_spiketrain, delta_modulation

using DSP   

"""
Perform delta-modulation on a real-valued signal.
The function generates a spike-train by emitting:
    - an upward spike `Spike(t, true)` when the signal increases
    by at least `Δ` relative to the last spike level
    - a downward spike `Spike(t, false)` when it decreases by at 
    least `Δ`
    
# Arguments
- `signal`: Time-series vector. 
- `Δ=100`: Threshold step size for emitting spikes

# Returns
- `Vector{Spike}`: A spike-train containing time-indexed spikes.
"""
function delta_modulation(
            signal::AbstractVector{T}; 
            Δ::Real=100
        ) where {T <: Real}
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

"""
Load raw ECG data for a given patient and session.

The function expects files stored as:

    `./ecg-db/patient{patient}/{session}.dat`

The `.dat` file is assumed to contain 16 interleaved `Int16` channels.
Channel 2 is extracted and returned as a `Vector{Float64}`.

# Arguments
- `patient`: Patient identifier used in folder naming.
- `session`: Session identifier corresponding to the `.dat` file.

# Returns
A `Vector{Float64}` containing samples from ECG channel 2.

# Throws
- `ErrorException` if the file does not exist.
"""
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


"""
Apply a 4th-order Butterworth bandpass filter to a signal.

# Arguments
- `signal`: Input time-series vector.
- `lowcut=0.01`: Lower cutoff frequency in Hz.
- `highcut=40`: Upper cutoff frequency in Hz.
- `fs=1000`: Sampling frequency in Hz.

# Returns
- Filtered signal vector with the same element type as `signal`.
"""
function get_filtered_signal(
            signal::AbstractVector{T}; 
            lowcut::Real=0.01, 
            highcut::Real=40, 
            fs::Real=1000) where {T <: Real}
    pass = Bandpass(lowcut, highcut)
    method = Butterworth(4)
    return filtfilt(digitalfilter(pass, method; fs=fs), signal)
end


"""
    get_spiketrain(patient, session; Δ=100)
        -> (spiketrain, signal_length, filtered_signal)

Load ECG data, apply bandpass filtering, and compute a delta-modulated spiketrain.

# Arguments
- `patient`: Patient identifier.
- `session`: Session identifier.
- `Δ=100`: Threshold parameter used in delta modulation.

# Returns
A tuple containing:
1. `spiketrain`: Encoded spike representation.
2. `signal_length`: Length of the filtered signal.
3. `filtered_signal`: Bandpass-filtered ECG signal.
"""
function get_spiketrain(patient, session; Δ=100)
    raw_sig = load_raw_signal(patient, session)
    filt_sig = get_filtered_signal(raw_sig)
    spiketrain = delta_modulation(filt_sig; Δ=Δ)

    return spiketrain, length(filt_sig), filt_sig
end

end # module Signals