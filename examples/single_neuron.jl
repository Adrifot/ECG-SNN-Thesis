include("../modules/Neurons.jl")

using .Neurons
using Plots

gr()

dt = 0.1
simtime = 100.0

volt_history = Float64[]

steps = 0:dt:simtime

const_input(t, amp, t0) = t > t0 ? amp : 0.0
inputfunc = (t) -> const_input(t, 2, 5.0)

n = Neuron("test_neuron"; τ_ref=3.0)

for t in steps
    n.i = inputfunc(t)
    fired = update!(n, dt, t)
    push!(volt_history, n.v)
end

p = plot(
    steps, volt_history,
    linecolor = :steelblue,
    linewidth = 2,
    title = "Single LIF Neuron Simulation",
    xlabel = "Time",
    ylabel = "Membrane Potential",
    label = "Voltage"
)

hline!([n.V_thresh], line=:dash, color=:red, label="Threshold")

display(p)
println("Press Enter...")
readline()
savefig(p, joinpath(@__DIR__, "../docs/imgs/single_neuron.png"))
println("Plot saved in docs/imgs/")

