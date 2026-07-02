include("../modules/Signals.jl")
include("../modules/Registry.jl")

using .Signals, .Registry
using Plots, Random
using Plots.Measures
using StatsBase: sample

gr()

const FS = 1000.0
const LEAD = 2
const Δ = 0.1          
const SEED = 3

# ----- get one clean normalized beat -----
Random.seed!(SEED)
rec = sample(filter(r -> r.label == :healthy, build_registry("./ecg-db")))
println("Using healthy patient $(rec.patient), session $(rec.session)")
sig   = get_filtered_signal(load_raw_signal(rec.patient, rec.session; lead=LEAD))
peaks = get_R_peaks(sig; fs=FS)
beats = segment_beats(sig, peaks; fs=FS)
beat  = normalize_beat(beats[max(1, length(beats) ÷ 2)])
t     = 1:length(beat)

# ----- delta modulation, recording the reference level at every step -----
function delta_with_levels(b, Δ)
    lvl = b[1]
    levels = fill(lvl, length(b))
    up_t = Int[]; dn_t = Int[]
    for i in 2:length(b)
        d = b[i] - lvl
        if d >= Δ
            lvl += Δ; push!(up_t, i)
        elseif d <= -Δ
            lvl -= Δ; push!(dn_t, i)
        end
        levels[i] = lvl
    end
    return levels, up_t, dn_t
end

levels, up_t, dn_t = delta_with_levels(beat, Δ)

# ----- top panel: beat, tracked level, and spike events -----
p1 = plot(t, beat; lw=2, color=:black, label="normalized beat",
          ylabel="amplitude", legend=:topright,
          title="Delta modulation (Δ = $(Δ))", titlefontsize=11)
plot!(p1, t, levels; lw=1.4, color=:gray, linestyle=:auto, label="tracked level ℓ")
scatter!(p1, up_t, beat[up_t]; markershape=:utriangle, color=:seagreen, ms=5,
         markerstrokewidth=0, label="up spike")
scatter!(p1, dn_t, beat[dn_t]; markershape=:dtriangle, color=:firebrick, ms=5,
         markerstrokewidth=0, label="down spike")

# ----- bottom panel: the resulting spike train (up = +1, down = -1) -----
p2 = plot(; ylim=(-1.6, 1.6), yticks=([-1,0,1], ["down","","up"]),
          xlabel="time (ms)", ylabel="spike", legend=false,
          title="Encoded spike train", titlefontsize=10)
for x in up_t; plot!(p2, [x,x], [0,1];  lw=1.6, color=:seagreen); end
for x in dn_t; plot!(p2, [x,x], [0,-1]; lw=1.6, color=:firebrick); end
hline!(p2, [0]; color=:black, lw=0.6)

fig = plot(p1, p2; layout=grid(2,1, heights=[0.68,0.32]), size=(880,520), margin=5mm)
out = joinpath(@__DIR__, "delta_modulation.png")
savefig(fig, out); println("Saved $(out)"); display(fig)
