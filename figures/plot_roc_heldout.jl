include("results_data.jl")
using Plots, Plots.Measures, Random
gr()

# ----- ROC points from scores + Bool labels (true = positive/infarction) -----
function roc_points(scores::AbstractVector{<:Real}, labels::AbstractVector{Bool})
    P = count(labels); N = count(!, labels)
    order = sortperm(scores; rev=true)         # high score first
    tp = 0; fp = 0
    fpr = Float64[0.0]; tpr = Float64[0.0]
    prev = Inf
    for i in order
        s = scores[i]
        if s != prev
            push!(fpr, fp / max(N,1)); push!(tpr, tp / max(P,1)); prev = s
        end
        labels[i] ? (tp += 1) : (fp += 1)
    end
    push!(fpr, fp / max(N,1)); push!(tpr, tp / max(P,1))   # (1,1)
    return fpr, tpr
end

function boot_auc_ci(scores, labels; nboot=2000, rng=MersenneTwister(SEED))
    n = length(scores); stats = Float64[]
    for _ in 1:nboot
        idx = rand(rng, 1:n, n)
        s = scores[idx]; l = labels[idx]
        (count(l) < 1 || count(!, l) < 1) && continue
        a = auroc(s, l); isnan(a) || push!(stats, a)
    end
    sort!(stats)
    lo = stats[max(1, floor(Int, 0.025*length(stats)))]
    hi = stats[max(1, ceil(Int, 0.975*length(stats)))]
    return lo, hi
end

r = heldout_scores()
fpr, tpr = roc_points(r.scores, r.labels)
lo, hi = boot_auc_ci(r.scores, r.labels)
auc = r.auc
println("Held-out AUROC = $(round(auc,digits=3))  95% CI [$(round(lo,digits=3)), $(round(hi,digits=3))]")

plt = plot(fpr, tpr; lw=2.5, color=:navy, legend=:bottomright,
           label="ROC (AUROC = $(round(auc, digits=3)))",
           xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           title="Held-out test ROC - routed multi-lead SNN", titlefontsize=11,
           xlim=(0,1), ylim=(0,1), size=(620, 560), margin=5mm, aspect_ratio=:equal)
plot!(plt, [0,1], [0,1]; ls=:dash, color=:gray, label="chance")
annotate!(plt, 0.62, 0.18, text("95% CI [$(round(lo,digits=3)), $(round(hi,digits=3))]", 9, :left, :gray30))

out = joinpath(@__DIR__, "roc_heldout.png")
savefig(plt, out); println("Saved $(out)")
