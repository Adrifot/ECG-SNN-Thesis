include("../modules/Signals.jl")
include("../modules/Registry.jl")

using .Signals, .Registry
using Plots, Random
using StatsBase: sample
using Plots.Measures

gr()  

const FS = 1000.0        
const LEAD = 2             
const SEG_SEC = 3.0           
const SKIP_SEC = 1.0           
const SEED = 7

Random.seed!(SEED)
records = filter(r -> r.label != :unknown, build_registry("./ecg-db"))
healthy = filter(r -> r.label == :healthy,    records)
infarction = filter(r -> r.label == :infarction, records)

h_rec = sample(healthy)
i_rec = sample(infarction)
println("Healthy:    patient $(h_rec.patient)  session $(h_rec.session)")
println("Infarction: patient $(i_rec.patient)  session $(i_rec.session)")

function segment(rec)
    sig  = get_filtered_signal(load_raw_signal(rec.patient, rec.session; lead=LEAD))
    lo   = round(Int, SKIP_SEC * FS) + 1
    hi   = min(length(sig), lo + round(Int, SEG_SEC * FS) - 1)
    seg  = sig[lo:hi]
    t    = (0:length(seg)-1) ./ FS
    return t, seg
end

th, sh = segment(h_rec)
ti, si = segment(i_rec)

ph = plot(th, sh; lw=1.3, color=:seagreen, legend=false,
          title="Healthy — patient $(h_rec.patient)", ylabel="amplitude",
          titlefontsize=10)
pi = plot(ti, si; lw=1.3, color=:firebrick, legend=false,
          title="Infarction — patient $(i_rec.patient)",
          xlabel="time (s)", ylabel="amplitude", titlefontsize=10)

fig = plot(ph, pi; margin=5mm, layout=(2,1), size=(900, 520),
           plot_title="Lead II, $(SEG_SEC)s segment (band-pass filtered)",
           plot_titlefontsize=11)

outpath = joinpath(@__DIR__, "healthy_vs_infarction.png")
savefig(fig, outpath)
println("Saved figure to $(outpath)")
display(fig)
