"""
    Signals.jl

Utilities for loading and preprocessing ECG recordings,
including bandpass filtering and delta modulation spike encoding.

# Provides
- `delta_modulation`: Delta-modulator function.
- `load_raw_signal`: ECG data loader function.
- `get_filtered_signal`: Signal filtering function.
- `get_spiketrain`: Signal to spiketrain convertor.

"""
module Signals

export load_raw_signal, get_filtered_signal, get_spiketrain, delta_modulation

include("Neurons.jl")
using .Neurons

using DSP   

"""
    delta_modulation(signal; Δ=100) -> Vector{T<:Real}

Perform delta-modulation on a real-valued signal.
The function generates a spike-train by emitting:
    - an upward spike `Spike(t, true, "source")` when the signal increases
    by at least `Δ` relative to the last spike level.
    - a downward spike `Spike(t, false, "source")` when it decreases by at 
    least `Δ`.
    
# Arguments
- `signal::AbstractVector{T<:Real}`: Time-series vector. 
- `Δ::Float64=100`: Threshold step size for emitting spikes

# Returns
- `Vector{Spike}`: A spike-train containing time-indexed spikes.
"""
function delta_modulation(
            signal::AbstractVector{T}; 
            Δ::Float64=100.0
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
    load_raw_signal(patient, session) -> Vector{Float64}

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
    get_filtered_signal(signal; lowcut, highcut, fs=1000) 
        -> Vector{T<:Real}

Apply a 4th-order Butterworth bandpass filter to a signal.

# Arguments
- `signal::AbstractVector{T<:Real}`: Input time-series vector.
- `lowcut::Float64=0.01`: Lower cutoff frequency in Hz.
- `highcut::Float64=40.0`: Upper cutoff frequency in Hz.
- `fs::Float64=1000.0`: Sampling frequency in Hz.

# Returns
- Filtered signal vector with the same element type as `signal`.
"""
function get_filtered_signal(
            signal::AbstractVector{T}; 
            lowcut::Float64=0.01, 
            highcut::Float64=40.0, 
            fs::Float64=1000.0) where {T <: Real}
    pass = Bandpass(lowcut, highcut)
    method = Butterworth(4)
    return filtfilt(digitalfilter(pass, method; fs=fs), signal)
end


"""
    get_spiketrain(patient, session; Δ=100)
        -> (spiketrain::Vector{Spike}, 
            signal_length::Integer, 
            filtered_signal::AbstractVector{T<:Real})
            

Load ECG data, apply bandpass filtering, and compute a delta-modulated spiketrain.

# Arguments
- `patient`: Patient identifier.
- `session`: Session identifier.
- `Δ::Float64=100.0`: Threshold parameter used in delta modulation.

# Returns
A tuple containing:
1. `spiketrain::Vector{Spike}`: Encoded spike representation.
2. `signal_length::Integer`: Length of the filtered signal.
3. `filtered_signal::Vector{T<:Real}`: Bandpass-filtered ECG signal.
"""
function get_spiketrain(patient, session; Δ::Float64=100.0)
    raw_sig = load_raw_signal(patient, session)
    filt_sig = get_filtered_signal(raw_sig)
    spiketrain = delta_modulation(filt_sig; Δ=Δ)

    return spiketrain, length(filt_sig), filt_sig
end

end # module Signals