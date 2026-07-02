include("../modules/Signals.jl")
include("../modules/Registry.jl")

using .Signals, .Registry
using Plots, Random
using Plots.Measures
using StatsBase: sample

gr()

const FS = 1000.0
const LEAD = 2
const SEED = 69
const PRE = round(Int, 0.30 * FS)   # ms before R 
const POST = round(Int, 0.45 * FS)   # ms after R 

ms(x) = round(Int, x * FS / 1000)     

# ----- pick a healthy recording and a clean middle beat -----
Random.seed!(SEED)
records = filter(r -> r.label == :healthy, build_registry("./ecg-db"))
rec = sample(records)
println("Using healthy patient $(rec.patient), session $(rec.session), lead II")

sig   = get_filtered_signal(load_raw_signal(rec.patient, rec.session))
peaks = get_R_peaks(sig; fs=FS)
valid = filter(p -> p - PRE >= 1 && p + POST <= length(sig), peaks)
R = valid[max(1, length(valid) ÷ 2)]

beat = sig[R-PRE : R+POST]
t_ms = (-PRE:POST)                      
r_idx = PRE + 1                         

# ----- locate the other waves by windowed extrema relative to R -----
argmin_in(lo, hi) = (w = beat[lo:hi]; lo + argmin(w) - 1)
argmax_in(lo, hi) = (w = beat[lo:hi]; lo + argmax(w) - 1)

q_idx = argmin_in(r_idx - ms(50),  r_idx)              # Q: min just before R
s_idx = argmin_in(r_idx,           r_idx + ms(60))     # S: min just after R
p_idx = argmax_in(r_idx - ms(250), r_idx - ms(100))    # P: max before R
t_idx = argmax_in(r_idx + ms(150), r_idx + ms(400))    # T: max in repolarisation

idx2ms(i) = t_ms[i]
pt(i) = (idx2ms(i), beat[i])

# ----- plot -----
plt = plot(t_ms, beat; lw=1.6, color=:black, legend=false,
           xlabel="time relative to R (ms)", ylabel="amplitude",
           title="PQRST complex (healthy beat, PTB lead II)", titlefontsize=11,
           size=(820, 460), margin=5mm)

# ST segment: from the J point (= S, end of QRS) to the onset of the T wave.
# T onset is found as the first point after S where the signal rises past
# a fraction of the way from the (post-S) baseline to the T peak.
baseline = beat[s_idx]
t_thresh = baseline + 0.5 * (beat[t_idx] - baseline)
t_onset_idx = s_idx + findfirst(i -> beat[i] >= t_thresh, s_idx:t_idx) - 1

j_ms = idx2ms(s_idx)
st_end_ms = idx2ms(t_onset_idx)
vspan!(plt, [j_ms, st_end_ms]; alpha=0.15, color=:orange, label="")
annotate!(plt, (j_ms + st_end_ms)/2, minimum(beat) + 0.05*(maximum(beat)-minimum(beat)),
          text("ST", 9, :darkorange))

# mark and label each wave
waves = [(p_idx,"P",:dodgerblue,:top), (q_idx,"Q",:purple,:bottom),
         (r_idx,"R",:red,:top), (s_idx,"S",:purple,:bottom), (t_idx,"T",:seagreen,:top)]
yr = maximum(beat) - minimum(beat)
for (i, lab, col, side) in waves
    x, y = pt(i)
    scatter!(plt, [x], [y]; color=col, markersize=5, markerstrokewidth=0)
    dy = side == :top ? 0.06*yr : -0.08*yr
    annotate!(plt, x, y + dy, text(lab, 11, col, :bold))
end

outpath = joinpath(@__DIR__, "pqrst_annotated.png")
savefig(plt, outpath)
println("Saved figure to $(outpath)")
display(plt)
