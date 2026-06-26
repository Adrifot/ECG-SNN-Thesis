using Plots
using Plots.Measures
gr()

# ----- STDP window params -----
const A_plus = 1.0 # potentiation amplitude
const A_minus = 0.8 # depression amplitude 
const τ_plus = 17.0 # ms, potentiation time constant
const τ_minus = 34.0 # ms, depression time constant

Δt = -100.0:0.5:100.0
Δw = map(Δt) do d
    d > 0  ?  A_plus  * exp(-d / τ_plus)  :
    d < 0  ? -A_minus * exp( d / τ_minus) :
             0.0
end

# split for two-colour fill (potentiation vs depression)
pos = [d > 0 ? w : NaN for (d, w) in zip(Δt, Δw)]
neg = [d < 0 ? w : NaN for (d, w) in zip(Δt, Δw)]

plt = plot(Δt, pos; lw=2.5, color=:seagreen, fillrange=0, fillalpha=0.25,
           label="potentiation (LTP)", legend=:topright,
           xlabel="Δt = t_post − t_pre  (ms)", ylabel="Δw",
           title="STDP learning window", titlefontsize=11,
           size=(760, 440), margin=5mm)
plot!(plt, Δt, neg; lw=2.5, color=:firebrick, fillrange=0, fillalpha=0.25,
      label="depression (LTD)")
hline!(plt, [0]; color=:black, lw=0.8, label="")
vline!(plt, [0]; color=:black, ls=:dash, lw=0.8, label="")

# quadrant annotations
annotate!(plt,  45,  0.55, text("pre before post\n→ strengthen", 9, :seagreen))
annotate!(plt, -45, -0.45, text("post before pre\n→ weaken", 9, :firebrick))

out = joinpath(@__DIR__, "stdp_window.png")
savefig(plt, out); println("Saved $(out)"); display(plt)
