"""
# TODO: module docstring
"""
module Metrics

using Statistics
using Random

export balanced_acc, ConfusionMatrix, gettotal, get_confusion_matrix, get_value_matrix

"""
    balanced_acc(TP, TN, FP, FN) -> Real
    balanced_acc(TPR, TNR) -> Real

Calculate accuracy accounted for imbalanced classes.
"""
function balanced_acc(TP::Real, TN::Real, FP::Real, FN::Real)
    # sensitivity = recall
    sensitivity = TP / (TP + FN + 1e-8) # add 1e-8 to avoid division by zero
    specificity = TN / (TN + FP + 1e-8)
    return (sensitivity + specificity) / 2
end

balanced_acc(TPR::Real, TNR::Real) = (TPR + TNR) / 2

"""
    ConfusionMatrix(TP. TN, FP, FN)

A matrix containing raw counts for binary classification tasks.

# Fields:
- `TP::Int`: True Positives
- `TN::Int`: True Negatives
- `FP::Int`: False Positives
- `FN::Int`: False Negatives
"""
struct ConfusionMatrix
    TP::Int
    TN::Int
    FP::Int
    FN::Int
end

gettotal(CM::ConfusionMatrix) = CM.TP + CM.TN + CM.FP + CM.FN

"""
    get_confusion_matrix(y_true, y_pred) -> ConfusionMatrix

Returns a confusion matrix based on two boolean vectors.
"""
function get_confusion_matrix(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
    length(y_true) == length(y_pred) || throw(ArgumentError("`y_true` and `y_pred` must have equal length."))
    TP = FN = FP = TN = 0
    @inbounds for i in eachindex(y_true)
        if y_true[i]
            y_pred[i] ? TP += 1 : FN += 1 
        else
            y_pred[i] ? FP += 1 : TN += 1
        end
    end
    return ConfusionMatrix(TP, TN, FP, FN)
end

"""
    get_value_matrix(ConfusionMatrix) -> Matrix{Int}

Returns the numerical values of a `ConfusionMatrix` struct.
"""
get_value_matrix(CM::ConfusionMatrix) = [CM.TP CM.FN; 
                                        CM.FP CM.TN]

"""
    mcc(ConfusionMatrix) -> Float64
    mcc(y_true, y_pred) -> Float64

Calculate Matthews Correlation Coefficient (MCC). 

# Returns
- `Float64 ∈ [-1, 1]`: the MCC value
"""
function mcc(CM::ConfusionMatrix)
    TP, FP, FN, TN = get_value_matrix(CM) # column-first matrix traversal
    num = (TP * TN) - (FP * FN)
    den = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return den < 1e-8 ? 0.0 : num/den
end

mcc(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool}) = mcc(get_confusion_matrix(y_true, y_pred))

precision_score(CM::ConfusionMatrix) = (CM.TP + CM.FP) == 0 ? 0.0 : CM.TP / (CM.TP + CM.FP)
recall_score(CM::ConfusionMatrix) = (CM.TP + CM.FN) == 0 ? 0.0 : CM.TP / (CM.TP + CM.FN)
specificity(CM::ConfusionMatrix) = (CM.TN + CM.FP) == 0 ? 0.0 : CM.TN / (CM.TN + CM.FP)
npv(CM::ConfusionMatrix) = (CM.TN + CM.FN) == 0 ? 0.0 : CM.TN / (CM.TN + CM.FN)

"""
    auroc(scores, labels) => Float64

Calculate the area under the ROC curve via the Mann-Whitney U statistic (which does not assume normality). 

# Behavior
    # TODO: add docstring section
"""
function auroc(scores::AbstractVector{<:Real}, labels::AbstractVector{Bool})
    length(scores) == length(labels) || throw(ArgumentError("Length mismatch"))
    n_pos = count(labels)
    n_neg = length(labels) - n_pos
    (n_pos == 0 || n_neg == 0) && return NaN
    order = sortperm(scores)
    ranks = Vector{Float64}(undef, length(scores))
    i = 1
    @inbounds while i <= length(order)
        j = i
        while j < length(order) && scores[order[j+1]] == scores[order[i]]
            j += 1
        end
        avg_rank = (i + j) / 2
        for k in i:j
            ranks[order[k]] = avg_rank
        end
        i = j + 1
    end
    sum_pos = 0.0
    @inbounds for i in eachindex(labels)
        labels[i] && (sum_pos += ranks[i])
    end
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
end

"""
    kl_divergence(P, Q)

    Calculate Kullback-Lieber divergence:
    KL(P || Q) = ∑ Pᵢ * log(Pᵢ / Qᵢ)
    where P and Q are probability vectors.
"""
function kl_divergence(P::AbstractVector{<:Real}, Q::AbstractVector{<:Real})
    length(P) == length(Q) || error("Probability vectors must have the same length")
    kl = 0.0
    for i in eachindex(P)
        Pᵢ = Float64(P[i])
        Qᵢ = Float64(Q[i])
        if Pᵢ > 0
            if Qᵢ > 0   
                kl += Pᵢ * log(Pᵢ / Qᵢ)
            else
                KL += Pᵢ & log(Pᵢ / 1e-8)
            end
        end
    end
    return kl
end

"""
    kl_estimate(samples A, samples B; bins=50)

Estimate KL(P || Q) from sample histograms with shared bin edges.
"""
function kl_estimate(samples_a::AbstractVector{<:Real}, samples_b::AbstractVector{<:Real}; bins::Int=50)
    lo = min(minimum(samples_a), minimum(samples_b))
    hi = max(maximum(samples_a), maximum(samples_b))
    edges = range(lo, hi; length=bins + 1)
    counts_a = [count(x -> edges[i] <= x < edges[i+1], samples_a) for i in 1:bins]
    counts_b = [count(x -> edges[i] <= x < edges[i+1], samples_b) for i in 1:bins]
    P = counts_a / (sum(counts_a) + 1e-8)
    Q = counts_b / (sum(counts_b) + 1e-8)
    return kl_divergence(P, Q)
end

end # module Metrics
