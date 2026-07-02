"""
    demo_healthy_vs_infarction.jl

Presentation demo. Runs the SAME routed-network configuration (winning
hyperparameters from `confirm_routed_multilead.jl`, capstone test AUROC
0.851) on a healthy and an infarction ECG recording, and renders a
side-by-side animation of:

  1. the delta-modulated input encoding (P/QRS/T, up/down pulses),
  2. the resulting hidden- and output-layer spike rasters,
  3. the hidden→output synaptic weight matrix evolving under STDP.

This visualizes what happens BEFORE any feature is handed to the LDA
classifier — it does not reproduce a classification/detection result on its
own (see `confirm_routed_multilead.jl` / `eval_routed_multilead.jl` for the
actual scored pipeline).
"""

include("../modules/Layers.jl")
include("../modules/Signals.jl")
include("../modules/Registry.jl")

using .Signals, .Layers, .Layers.Neurons, .Layers.Synapses, .Layers.Utils, .Registry
using Plots, Random, Statistics
using StatsBase: sample

gr()

# ----- Winning params (from confirm_routed_multilead.jl) -----
const Δ            = 0.1
const pulse_amp    = 170.48121762522024
const ltp_rate     = 0.18935145456196095
const ltp_rate_out = 0.08378635445090134
const τ_s          = 42.831648889100016
const τ_s_output   = 25.003717150125617
const τ_pretrace   = 34.510086127922726
const τ_posttrace  = 47.00571781147488
const R_m_input    = 1.0674163386846072
const R_m_output   = 2.8737140296290087
const τ_m_input    = 9.41794217755773
const τ_m_output   = 2.4038829712094394
const τ_ref_input  = 7.209849285084017
const τ_ref_output = 0.5566821677223849
const N_in         = 20
const N_hid        = 25
const N_out        = 25
const density      = 0.7793537471176322
const inhib_den    = 0.2926531513089269
const inhib_str    = 0.05207908906117163

# ----- demo timing / rendering config -----
const dt          = 0.001
const FS          = 1000.0
const LEAD        = 2          # lead II
const PRE_R       = 0.25
const GAP         = 100.0
const R_IDX       = round(Int, PRE_R * FS) + 1
const TSIM        = 7.0        # seconds of network time (kept short: past ~8s both
                                # conditions saturate to a fully-LTD-collapsed weight
                                # matrix regardless of input, which is uninteresting)
const SNAPSHOT_DT = 0.025      # animation frame spacing
const RASTER_WIN  = 2.0        # trailing window shown in the scrolling rasters
const FPS         = 15
const NET_SEED    = 20240      # identical init for both conditions -> fair comparison
const PICK_SEED   = 7

seg_base(t) = t < R_IDX - 60 ? 0 : (t <= R_IDX + 80 ? 2 : 4)

# ----- network (same architecture/params as confirm_routed_multilead.jl) -----
function routed_net(rng)
    inp(rev) = Neuron(rev ? "id" : "iu"; R_m=R_m_input, τ_m=τ_m_input, τ_s=τ_s, τ_ref=τ_ref_input,
                       τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace, isreverse=rev)
    hidt = Neuron("h"; R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output,
                   τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    outt = Neuron("o"; R_m=R_m_output, τ_m=τ_m_output, τ_s=τ_s_output, τ_ref=τ_ref_output,
                   τ_pretrace=τ_pretrace, τ_posttrace=τ_posttrace)
    exc = Synapse(1, 2; learningrate=ltp_rate, wmax=1.0)
    exc_out = Synapse(1, 2; learningrate=ltp_rate_out, wmax=1.0)
    inh = Synapse(1, 1; learningrate=0.0, wmax=1.0, isinhibitory=true)
    ml(t, n, nm) = NeuronLayer(n, t; name=nm, V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15, rng=rng)
    ins = [ml(inp(iseven(i)), N_in, "in$(i)") for i in 1:6]
    hd = ml(hidt, N_hid, "h")
    ou = ml(outt, N_out, "o")
    syns = SynapseLayer[]
    for i in 1:6
        push!(syns, SynapseLayer(ins[i], hd, exc; dist=NormalDist(0.5, 0.2), density=density, pre_idx=i, post_idx=7, rng=rng))
    end
    push!(syns, SynapseLayer(hd, ou, exc_out; dist=NormalDist(0.5, 0.2), density=density, pre_idx=7, post_idx=8, rng=rng))
    idist = UniformDist(0.05, inhib_str)
    for i in 1:6
        push!(syns, SynapseLayer(ins[i], ins[i], inh; dist=idist, density=inhib_den, pre_idx=i, post_idx=i, rng=rng))
    end
    push!(syns, SynapseLayer(hd, hd, inh; dist=idist, density=inhib_den, pre_idx=7, post_idx=7, rng=rng))
    push!(syns, SynapseLayer(ou, ou, inh; dist=idist, density=inhib_den, pre_idx=8, post_idx=8, rng=rng))
    return LayeredNetwork(vcat(ins, [hd, ou]), syns)
end

function delta_mod(beat)
    n = length(beat); lvl = beat[1]; out = Spike[]
    for t in 2:n
        d = beat[t] - lvl
        if d >= Δ
            push!(out, Spike(Float64(t), true, "d")); lvl += Δ
        elseif d <= -Δ
            push!(out, Spike(Float64(t), false, "d")); lvl -= Δ
        end
    end
    return out
end

function routed_pulses(rec, lead, nsteps)
    filt = get_filtered_signal(load_raw_signal(rec.patient, rec.session; lead=lead))
    beats = segment_beats(filt, get_R_peaks(filt; fs=FS); fs=FS)
    arr = [zeros(nsteps) for _ in 1:6]
    offset = 0.0
    for beat in beats
        for s in delta_mod(normalize_beat(beat))
            step = Int(floor(s.time + offset)) + 1
            1 <= step <= nsteps || continue
            base = seg_base(s.time)
            s.polarity ? (arr[base+1][step] += pulse_amp) : (arr[base+2][step] -= pulse_amp)
        end
        offset += length(beat) + GAP
    end
    return arr
end

# ----- record selection -----
Random.seed!(PICK_SEED)
records    = filter(r -> r.label != :unknown, build_registry("./ecg-db"))
healthy    = filter(r -> r.label == :healthy,    records)
infarction = filter(r -> r.label == :infarction, records)
h_rec = sample(healthy)
i_rec = sample(infarction)
println("Healthy:    patient $(h_rec.patient)  session $(h_rec.session)")
println("Infarction: patient $(i_rec.patient)  session $(i_rec.session)")

# ----- run one condition, recording weight & spike history over time -----
function simulate(rec, label)
    nsteps = Int(round(TSIM / dt))
    arr = routed_pulses(rec, LEAD, nsteps)
    input_times = [(findall(!=(0.0), arr[g]) .- 1) .* dt for g in 1:6]

    net = routed_net(MersenneTwister(NET_SEED))
    hid_out = net.synapselayers[7]  # hidden(7) -> output(8), excitatory

    input_fn = t -> begin
        idx = clamp(Int(floor(t / dt)) + 1, 1, nsteps)
        v = zeros(length(net.neuronlayers))
        @inbounds for k in 1:6
            v[k] = arr[k][idx]
        end
        v
    end

    weight_times   = Float64[]
    weight_history = Matrix{Float64}[]
    hidden_events  = Tuple{Float64,Int}[]
    output_events  = Tuple{Float64,Int}[]

    last_hid = fill(-Inf, N_hid)
    last_out = fill(-Inf, N_out)
    next_snap = 0.0

    cb = function(t, net, step)
        if t >= next_snap - 1e-9
            hd = net.neuronlayers[7]; ou = net.neuronlayers[8]
            for n in 1:N_hid
                hd.t_lastout[n] > last_hid[n] && push!(hidden_events, (hd.t_lastout[n], n))
            end
            for n in 1:N_out
                ou.t_lastout[n] > last_out[n] && push!(output_events, (ou.t_lastout[n], n))
            end
            last_hid .= hd.t_lastout
            last_out .= ou.t_lastout
            push!(weight_times, t)
            push!(weight_history, copy(hid_out.ws))
            next_snap += SNAPSHOT_DT
        end
    end

    runlayers!(net, dt, TSIM; inputfn=input_fn, callback=cb)
    println("[$label] final hidden→output weight: mean=$(round(mean(hid_out.ws), digits=3)) " *
            "std=$(round(std(hid_out.ws), digits=3))  |  output spikes=$(length(output_events))")

    return (input_times=input_times, weight_times=weight_times, weight_history=weight_history,
            hidden_events=hidden_events, output_events=output_events)
end

res_h = simulate(h_rec, "healthy")
res_i = simulate(i_rec, "infarction")

# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
default(grid=false, titlefontsize=10, guidefontsize=8, tickfontsize=7, legend=false, framestyle=:box)

const GROUP_LABELS = ["P+", "P-", "QRS+", "QRS-", "T+", "T-"]
const GROUP_COLORS = [:seagreen, :seagreen, :steelblue, :steelblue, :darkorange, :darkorange]

function input_raster(input_times, t; win=RASTER_WIN)
    p = plot(xlim=(t - win, t), ylim=(0.5, 6.5), yticks=(1:6, GROUP_LABELS))
    for g in 1:6
        times = filter(x -> t - win <= x <= t, input_times[g])
        isempty(times) && continue
        scatter!(p, times, fill(g, length(times)); ms=2.2, mc=GROUP_COLORS[g], msw=0)
    end
    vline!(p, [t]; color=:black, ls=:dot, lw=1)
    return p
end

function unit_raster(hidden_events, output_events, t; win=RASTER_WIN)
    p = plot(xlim=(t - win, t), ylim=(0.5, N_hid + N_out + 0.5), xlabel="time (s)",
             yticks=([N_hid / 2, N_hid + N_out / 2], ["hidden", "output"]))
    hid = [(tm, n) for (tm, n) in hidden_events if t - win <= tm <= t]
    out = [(tm, n + N_hid) for (tm, n) in output_events if t - win <= tm <= t]
    isempty(hid) || scatter!(p, first.(hid), last.(hid); ms=2.0, mc=:gray40, msw=0)
    isempty(out) || scatter!(p, first.(out), last.(out); ms=2.4, mc=:crimson, msw=0)
    hline!(p, [N_hid + 0.5]; color=:black, lw=0.5, ls=:dash)
    vline!(p, [t]; color=:black, ls=:dot, lw=1)
    return p
end

weight_heatmap(W; title="") = heatmap(W; clims=(0, 1), color=:viridis, colorbar=false, title=title,
                                       xlabel="hidden idx", ylabel="output idx", yflip=true)

n_frames = length(res_h.weight_times)
println("\nRendering $(n_frames) frames…")

anim = @animate for k in 1:n_frames
    t = res_h.weight_times[k]

    ph1 = input_raster(res_h.input_times, t)
    ph2 = unit_raster(res_h.hidden_events, res_h.output_events, t)
    ph3 = weight_heatmap(res_h.weight_history[k]; title="Healthy — patient $(h_rec.patient)")

    pi1 = input_raster(res_i.input_times, t)
    pi2 = unit_raster(res_i.hidden_events, res_i.output_events, t)
    pi3 = weight_heatmap(res_i.weight_history[k]; title="Infarction — patient $(i_rec.patient)")

    plot(ph1, pi1, ph2, pi2, ph3, pi3, layout=(3, 2), size=(1000, 780),
         plot_title="t = $(round(t, digits=2))s / $(TSIM)s", plot_titlefontsize=11)
end

imgdir = joinpath(@__DIR__, "../docs/imgs")
isdir(imgdir) || mkpath(imgdir)

gifpath = joinpath(imgdir, "demo_healthy_vs_infarction.gif")
gif(anim, gifpath, fps=FPS)
println("Saved animation to $(gifpath)")

# Opt-in: dump the individual frames (reused from the animation, no extra render
# cost) so the Beamer deck can embed them via \animategraphics. Enable with:
#   DEMO_FRAMES=1 julia --project=. examples/demo_healthy_vs_infarction.jl
if get(ENV, "DEMO_FRAMES", "0") == "1"
    framedir = joinpath(@__DIR__, "../docs/presentation/frames")
    isdir(framedir) ? foreach(f -> rm(joinpath(framedir, f)), readdir(framedir)) : mkpath(framedir)
    for (i, f) in enumerate(anim.frames)
        cp(joinpath(anim.dir, f), joinpath(framedir, "frame-$(lpad(i-1, 3, '0')).png"); force=true)
    end
    println("Saved $(length(anim.frames)) frames to $(framedir) (fps=$(FPS))")
end

final = plot(
    weight_heatmap(res_h.weight_history[end]; title="Healthy — final weights"),
    weight_heatmap(res_i.weight_history[end]; title="Infarction — final weights"),
    layout=(1, 2), size=(800, 350))
pngpath = joinpath(imgdir, "demo_healthy_vs_infarction_final.png")
savefig(final, pngpath)
println("Saved final-frame preview to $(pngpath)")
