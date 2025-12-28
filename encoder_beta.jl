using Pkg
using CondaPkg
using PythonCall
using Plots
using DSP

gr()

wfdb = pyimport("wfdb") # import the Python library

record_path = "./ecg-patient1/s0010_re"

# record.p_signal contains the values in physical units (mV)
# record.sig_name contains the lead names (I, II, III, V1, VX, VY, VZ, etc.)
record = wfdb.rdrecord(record_path)

signals = pyconvert(Array, record.p_signal)
lead_names = pyconvert(Vector{String}, record.sig_name)
fs = pyconvert(Int, record.fs) # Sampling freq

println("Loaded $(length(lead_names)) leads at $(fs)Hz.")

sec_start = 1
sec_end = 2
idx = 1:fs*sec_end
lead = 2


time_ax = (0:size(signals, 1)-1) ./ fs

low_cut = 0.75
high_cut = 35.0

pass = Bandpass(low_cut, high_cut)
method = Butterworth(4)
my_filter = digitalfilter(pass, method; fs=fs)

signal_filtered = filtfilt(my_filter, signals[:, lead])

# p_comp = plot(
#     plot(time_ax[idx], signals[idx, lead], title="UNFILTERED", ylabel="mV", color=:gray),
#     plot(time_ax[idx], signal_filtered[idx], title="FILTERED", ylabel="mV", color=:blue),
#     layout = (2, 1), # 2 rows, 1 column
#     size = (900, 700),
#     legend = false
# )

# display(p_comp)
# println("Press ENTER to exit.")
# readline()

function delta_modulation(signal; Δ=0.1)
    up = []
    down = []
    # 'last_spike_level' only updates when we cross the threshold
    last_spike_level = signal[1] 
    
    for i ∈ 2:length(signal)
        diff = signal[i] - last_spike_level
        
        if diff >= Δ
            push!(up, i)
            last_spike_level = signal[i] # Update reference
        elseif diff <= -Δ
            push!(down, i)
            last_spike_level = signal[i] # Update reference
        end
        # If neither condition is met, we do NOTHING. 
        # This creates the "white space" between spikes.
    end
    return up, down
end

up_idx, down_idx = delta_modulation(signal_filtered[idx], Δ=0.1)
p_spike = plot(time_ax[idx], signal_filtered[idx], color=:black, label="Filtered ECG")
scatter!(time_ax[idx][up_idx], signal_filtered[idx][up_idx], 
         color=:red, markersize=3, label="UP Spike")
scatter!(time_ax[idx][down_idx], signal_filtered[idx][down_idx], 
         color=:blue, markersize=3, label="DOWN Spike")

display(p_spike)
println("Press ENTER to exit.")
readline()