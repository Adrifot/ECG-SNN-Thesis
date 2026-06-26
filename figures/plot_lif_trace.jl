using Plots
using Plots.Measures
gr()

# ----- LIF parameters -----
const dt = 0.1          
const T = 120.0        
const V_rest = 0.0
const V_thresh = 1.0
const V_reset = 0.0
const R_m = 1.0
const τ_m = 10.0         
const τ_ref = 4.0          
const I_in = 1.5          
const V_SPIKE = 1.5          

nsteps = round(Int, T / dt)
t = collect(0:nsteps-1) .* dt

function simulate_lif()
    V = fill(V_rest, nsteps)
    spikes = Float64[]
    ref_windows = Tuple{Float64,Float64}[]
    v = V_rest
    t_ref = 0.0
    for k in 2:nsteps
        if t_ref > 0
            v = V_reset
            t_ref -= dt
        else
            v += dt/τ_m * (-(v - V_rest) + R_m * I_in)
            if v >= V_thresh
                push!(spikes, t[k])
                push!(ref_windows, (t[k], t[k] + τ_ref))
                v = V_reset
                t_ref = τ_ref
            end
        end
        V[k] = v
    end
    return V, spikes, ref_windows
end

V, spikes, ref_windows = simulate_lif()

# Draw spikes as vertical lines up to V_SPIKE for a recognisable trace
plt = plot(t, V; lw=2, color=:navy, legend=:topright, label="membrane potential V(t)",
           xlabel="time (ms)", ylabel="V", ylim=(-0.25, V_SPIKE + 0.15),
           title="Leaky integrate-and-fire neuron", titlefontsize=11,
           size=(820, 420), margin=5mm)
for (i, ts) in enumerate(spikes)
    plot!(plt, [ts, ts], [V_reset, V_SPIKE]; lw=2, color=:navy, label=(i==1 ? "spike" : ""))
end
for (i, (a, b)) in enumerate(ref_windows)
    vspan!(plt, [a, b]; alpha=0.10, color=:gray, label=(i==1 ? "refractory" : ""))
end
hline!(plt, [V_thresh]; ls=:dash, color=:firebrick, label="V_thresh")
hline!(plt, [V_reset];  ls=:dot,  color=:gray,      label="V_reset")

out = joinpath(@__DIR__, "lif_trace.png")
savefig(plt, out); println("Saved $(out)"); display(plt)
