"""
    Signals.jl

Utilities for loading and preprocessing ECG recordings,
including bandpass filtering and delta modulation spike encoding.

# Provides #TODO: add newly added functions
- `delta_modulation`: Delta-modulator function.
- `load_raw_signal`: ECG data loader function.
- `get_filtered_signal`: Signal filtering function.
- `get_spiketrain`: Signal to spiketrain convertor.

"""
module Signals

export get_spiketrain, load_raw_signal, get_filtered_signal, get_R_peaks, segment_beats, normalize_beat, delta_modulation

include("Neurons.jl")
using .Neurons

using DSP   
using Statistics

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
            Δ::Float64=0.1
        ) where {T <: Real}
    n = length(signal)
    last_spike_lvl = signal[1]
    spiketrain = Spike[]
    for t in 2:n
        diff = signal[t] - last_spike_lvl 
        if diff >= Δ
            push!(spiketrain, Spike(Float64(t), true, "delta"))
            last_spike_lvl += Δ 
        elseif diff <= -Δ
            push!(spiketrain, Spike(Float64(t), false, "delta"))
            last_spike_lvl -= Δ
        end
    end
    return spiketrain
end

"""
    load_raw_signal(patient, session; lead=2) -> Vector{Float64}

Load raw ECG data for a given patient and session.

The function expects files stored as:

    `./ecg-db/patient{patient_id}/{session_id}.dat`

The `.dat` file holds the standard 12 leads as interleaved `Int16` channels.
The requested lead is extracted and returned as a `Vector{Float64}`.

# Arguments
- `patient`: Patient identifier used in folder naming.
- `session`: Session identifier corresponding to the `.dat` file.
- `lead=2`: Lead to extract. Either a 1-based channel index (e.g. `2`) or a
  lead name matching the header label, as a `String` or `Symbol`
  (e.g. `"ii"`, `:v1`); name matching is case-insensitive.

# Returns
A `Vector{Float64}` containing samples from the requested lead.

# Throws
- `ErrorException` if a file is missing, the lead name is not found, or the
  channel index is out of range.
"""
function load_raw_signal(patient, session; lead::Union{Integer,AbstractString,Symbol}=2)
    dat_path = "./ecg-db/patient$(patient)/$(session).dat"
    hea_path = "./ecg-db/patient$(patient)/$(session).hea"

    !isfile(dat_path) && error("File not found: $(dat_path)")
    !isfile(hea_path) && error("Header not found: $(hea_path)")

    lines = readlines(hea_path)
    n_samples = parse(Int, split(lines[1])[4])

    dat_name = basename(dat_path)
    dat_specs = filter(l -> !isempty(l) && first(split(l)) == dat_name, lines[2:end])
    n_channels = length(dat_specs)
    lead_names = [last(split(l)) for l in dat_specs]

    # resolve the requested lead to a 1-based channel index
    if lead isa Integer
        ch = Int(lead)
        (1 ≤ ch ≤ n_channels) ||
            error("Lead index $(ch) out of range 1:$(n_channels) for $(dat_name)")
    else
        name = lowercase(String(lead))
        ch = findfirst(==(name), lowercase.(lead_names))
        if ch === nothing
            available = join(lead_names, ", ")
            error("Lead \"$(lead)\" not found in $(dat_name); available: $(available)")
        end
    end

    raw_data = reinterpret(Int16, read(dat_path))
    expected = n_channels * n_samples

    if length(raw_data) != expected
        usable = (length(raw_data) ÷ n_channels) * n_channels
        raw_data = raw_data[1:usable]
    end

    data_matrix = reshape(raw_data, n_channels, :)
    return Float64.(data_matrix[ch, :])
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
            lowcut::Float64=0.5, 
            highcut::Float64=40.0, 
            fs::Float64=1000.0) where {T <: Real}
    pass = Bandpass(lowcut, highcut)
    method = Butterworth(4)
    return filtfilt(digitalfilter(pass, method; fs=fs), signal)
end


"""
# TODO: docstring
"""
function get_spiketrain(patient, session; Δ::Float64=0.1, fs::Float64=1000.0, gap::Float64=100.0)
    raw_sig = load_raw_signal(patient, session)
    filt_sig = get_filtered_signal(raw_sig)
    peaks = get_R_peaks(filt_sig; fs=fs)
    beats = segment_beats(filt_sig, peaks; fs=fs)
    beats_norm = normalize_beat.(beats)

    spiketrain = Spike[]
    offset = 0.0
    for beat in beats_norm
        beat_spikes = delta_modulation(beat; Δ=Δ)
        for s in beat_spikes
            push!(spiketrain, Spike(s.time + offset, s.polarity, s.src_name))
        end
        offset += length(beat) + gap
    end
    return spiketrain, length(filt_sig), filt_sig
end

"""
#TODO: docstring
"""
function get_R_peaks(
            signal::AbstractVector{T}; 
            fs::Float64=1000.0,
            min_d::Int=100) where {T <: Real}

    sig = get_filtered_signal(signal; lowcut=5.0, highcut=15.0)

    diffsig = [0.0; diff(sig)]
    squaredsig = diffsig .^ 2
    window = round(Int, 0.15*fs)
    window = window + (iseven(window) ? 1 : 0)
    
    kernel = ones(window) / window
    smoothed = filt(kernel, [1.0], squaredsig)

    thresh = mean(smoothed) + 0.5 * std(smoothed)
    candidates = findall(smoothed .> thresh)

    peaks = Int[]
    i = 1
    while i ≤ length(candidates)
        start = i
        while i < length(candidates) && candidates[i+1] - candidates[i] < min_d
            i += 1
        end
        cluster = candidates[start:i]
        left  = max(1, first(cluster) - round(Int, 0.1*fs))
        right = min(length(sig), last(cluster) + round(Int, 0.1*fs))
        _, idx = findmax(sig[left:right])
        push!(peaks, left + idx - 1)
        i += 1
    end
    return peaks
end

"""
# TODO: docstring
"""
function segment_beats(
            signal::AbstractVector{T}, 
            peaks::Vector{Int};
            fs::Float64 = 1000.0, 
            pre_r::Float64 = 0.25, 
            post_r::Float64 = 0.45) where {T <: Real}

    pre = round(Int, pre_r*fs)
    post = round(Int, post_r*fs)
    beats = Vector{Float64}[]
    
    for r in peaks
        startidx = r - pre
        endidx = r + post
        if startidx ≥ 1 && endidx ≤ length(signal)
            push!(beats, signal[startidx:endidx])
        end
    end
    return beats
end

"""
# TODO: docstring
"""
function normalize_beat(beat::Vector{Float64})
    low, high = minimum(beat), maximum(beat)
    r = high - low
    r < eps() && return zeros(length(beat))
    return @. (beat - low) / r
end

end # module Signals