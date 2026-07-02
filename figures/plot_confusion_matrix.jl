include("results_data.jl")
using Plots, Plots.Measures
gr()

r  = heldout_scores()
cc = confusion_counts(r.labels, r.scores .>= 0.0)

#   [ TN  FP ]
#   [ FN  TP ]

M = [cc.tn cc.fp;
     cc.fn cc.tp]
ba  = balanced_acc(cc); mc = mcc(cc); acc = accuracy(cc)
println("TN=$(cc.tn) FP=$(cc.fp) FN=$(cc.fn) TP=$(cc.tp) | bal.acc=$(round(100*ba,digits=1))% MCC=$(round(mc,digits=3))")

classes = ["Healthy", "Infarction"]
plt = heatmap(1:2, 1:2, M;
              c=:blues, clims=(0, maximum(M)), colorbar=false,
              xticks=(1:2, classes), yticks=(1:2, classes),
              xlabel="Predicted", ylabel="Actual", yflip=true,
              title="Held-out confusion matrix  (bal. acc. $(round(100*ba,digits=1))%, MCC $(round(mc,digits=2)))",
              titlefontsize=10, size=(560, 500), margin=6mm)

mx = maximum(M)
for i in 1:2, j in 1:2
    val = M[i, j]
    col = val > 0.55*mx ? :white : :black
    annotate!(plt, j, i, text(string(val), 16, col, :center))
end

out = joinpath(@__DIR__, "confusion_matrix.png")
savefig(plt, out); println("Saved $(out)")
