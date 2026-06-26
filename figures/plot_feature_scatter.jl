include("results_data.jl")
using Plots, Plots.Measures, LinearAlgebra
gr()

_, _, Xtest, ytest = build_feature_cache()

μ  = vec(mean(Xtest, dims=1))
Xc = Xtest .- μ'
F  = svd(Xc)
PC = Xc * F.V[:, 1:2]    # n × 2 scores
ev = (F.S .^ 2) ./ sum(F.S .^ 2)  # variance ratio
println("PC1 $(round(100*ev[1],digits=1))%  PC2 $(round(100*ev[2],digits=1))% of variance")

h = .!ytest; m = ytest
plt = scatter(PC[h, 1], PC[h, 2]; label="Healthy", color=:seagreen, ms=6, msw=0.5, ma=0.8,
              xlabel="PC1 ($(round(100*ev[1],digits=1))%)", ylabel="PC2 ($(round(100*ev[2],digits=1))%)",
              title="PCA of held-out firing-rate features", titlefontsize=11,
              legend=:best, size=(680, 560), margin=5mm)
scatter!(plt, PC[m, 1], PC[m, 2]; label="Infarction", color=:firebrick, ms=6, msw=0.5, ma=0.8,
         marker=:diamond)

out = joinpath(@__DIR__, "feature_scatter.png")
savefig(plt, out); println("Saved $(out)")
