"""
    ClassificationMetrics.jl

Imbalance-aware classification metrics for the ECG anomaly-detection network.

Designed for a trial-level evaluation in which each hybrid recording is one
sample: a healthy→infarction hybrid is a POSITIVE (anomaly present) and a
healthy→healthy hybrid is a NEGATIVE (no anomaly). The network's binary
"alarm fired?" output is the prediction.

All functions are pure (no simulation/IO) so they can be unit-tested in
isolation and reused from any runner script.

Provides:
- `confusion_counts`, `ConfusionMatrix`           — raw 2×2 counts
- `mcc`                                            — Matthews correlation coefficient
- `precision_score`, `recall_score`, `specificity`, `npv`
- `per_class_precision_recall`                     — precision/recall for BOTH classes
- `f1_score`, `fbeta_score`, `gmean`, `youden_j`, `cohens_kappa`
- `balanced_acc`, `accuracy`
- `wilson_ci`                                      — CI for a single proportion
- `bootstrap_ci`                                   — percentile CI for any trial-level metric
- `auroc`, `average_precision`                     — threshold-free ranking metrics
- `summarize`                                      — NamedTuple of everything at once
"""
module ClassificationMetrics

using Statistics
using Random

export ConfusionMatrix, confusion_counts, confusion_matrix,
       mcc, precision_score, recall_score, specificity, npv,
       per_class_precision_recall, f1_score, fbeta_score, gmean,
       youden_j, cohens_kappa, balanced_acc, accuracy,
       wilson_ci, bootstrap_ci, auroc, average_precision, summarize

const EPS = 1e-12

"""
    ConfusionMatrix

Raw counts for binary classification with `true` = positive (anomaly).
`tp` true positives, `fn` false negatives, `fp` false positives, `tn` true negatives.
"""
struct ConfusionMatrix
    tp::Int
    fn::Int
    fp::Int
    tn::Int
end

n_total(c::ConfusionMatrix) = c.tp + c.fn + c.fp + c.tn

"""
    confusion_counts(y_true, y_pred) -> ConfusionMatrix

`y_true` and `y_pred` are equal-length Boolean vectors; `true` = positive
(anomaly present / alarm fired).
"""
function confusion_counts(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
    length(y_true) == length(y_pred) ||
        throw(ArgumentError("y_true and y_pred must have equal length"))
    tp = fn = fp = tn = 0
    @inbounds for i in eachindex(y_true)
        if y_true[i]
            y_pred[i] ? (tp += 1) : (fn += 1)
        else
            y_pred[i] ? (fp += 1) : (tn += 1)
        end
    end
    return ConfusionMatrix(tp, fn, fp, tn)
end

"""
    confusion_matrix(y_true, y_pred) -> Matrix{Int}

2×2 layout (rows = actual, cols = predicted):

                 pred +    pred -
    actual +   [  tp        fn  ]
    actual -   [  fp        tn  ]
"""
function confusion_matrix(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
    c = confusion_counts(y_true, y_pred)
    return [c.tp c.fn; c.fp c.tn]
end

"""
    mcc(c) -> Float64

Matthews correlation coefficient in [-1, 1]. Robust to class imbalance:
1 = perfect, 0 = no better than chance, -1 = total disagreement. Returns 0.0
when any margin is empty (the standard convention).
"""
function mcc(c::ConfusionMatrix)
    tp, fn, fp, tn = c.tp, c.fn, c.fp, c.tn
    num = (tp * tn) - (fp * fn)
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return den < EPS ? 0.0 : num / den
end
mcc(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool}) = mcc(confusion_counts(y_true, y_pred))

"Positive predictive value (precision) for the positive/anomaly class."
precision_score(c::ConfusionMatrix) = (c.tp + c.fp) == 0 ? 0.0 : c.tp / (c.tp + c.fp)

"Sensitivity / true-positive rate / recall for the positive/anomaly class."
recall_score(c::ConfusionMatrix) = (c.tp + c.fn) == 0 ? 0.0 : c.tp / (c.tp + c.fn)

"True-negative rate for the negative/healthy class."
specificity(c::ConfusionMatrix) = (c.tn + c.fp) == 0 ? 0.0 : c.tn / (c.tn + c.fp)

"Negative predictive value."
npv(c::ConfusionMatrix) = (c.tn + c.fn) == 0 ? 0.0 : c.tn / (c.tn + c.fn)

"""
    per_class_precision_recall(c) -> NamedTuple

Precision and recall for BOTH classes (positive = anomaly, negative = healthy),
the way `sklearn.classification_report` presents them.
"""
function per_class_precision_recall(c::ConfusionMatrix)
    pos_precision = precision_score(c)
    pos_recall    = recall_score(c)
    neg_precision = npv(c)
    neg_recall    = specificity(c)
    return (
        anomaly = (precision = pos_precision, recall = pos_recall,
                   support = c.tp + c.fn),
        healthy = (precision = neg_precision, recall = neg_recall,
                   support = c.tn + c.fp),
    )
end

"F_beta score (β>1 weights recall higher — usually what you want in screening)."
function fbeta_score(c::ConfusionMatrix; beta::Real = 1.0)
    p = precision_score(c)
    r = recall_score(c)
    b2 = beta^2
    den = b2 * p + r
    return den < EPS ? 0.0 : (1 + b2) * p * r / den
end

"F1 = harmonic mean of precision and recall (positive class)."
f1_score(c::ConfusionMatrix) = fbeta_score(c; beta = 1.0)

"Geometric mean of sensitivity and specificity — a strong single imbalance metric."
gmean(c::ConfusionMatrix) = sqrt(max(0.0, recall_score(c) * specificity(c)))

"Youden's J = sensitivity + specificity − 1, in [-1, 1]."
youden_j(c::ConfusionMatrix) = recall_score(c) + specificity(c) - 1.0

"Balanced accuracy = (sensitivity + specificity) / 2."
balanced_acc(c::ConfusionMatrix) = (recall_score(c) + specificity(c)) / 2

"Plain accuracy = (tp + tn) / n. Reported for reference only; misleading under imbalance."
accuracy(c::ConfusionMatrix) = n_total(c) == 0 ? 0.0 : (c.tp + c.tn) / n_total(c)

"""
    cohens_kappa(c) -> Float64

Cohen's κ: agreement corrected for chance. 1 = perfect, 0 = chance.
"""
function cohens_kappa(c::ConfusionMatrix)
    n = n_total(c)
    n == 0 && return 0.0
    po = (c.tp + c.tn) / n
    p_pos = ((c.tp + c.fn) / n) * ((c.tp + c.fp) / n)
    p_neg = ((c.tn + c.fp) / n) * ((c.tn + c.fn) / n)
    pe = p_pos + p_neg
    return (1 - pe) < EPS ? 0.0 : (po - pe) / (1 - pe)
end

"""
    wilson_ci(k, n; z=1.96) -> (lo, hi)

Wilson score interval for a binomial proportion k/n. Far better than the normal
approximation when n is small or the proportion is near 0 or 1 — appropriate
for sensitivity/specificity reported on a handful of trials. Default z=1.96 ⇒ 95%.
"""
function wilson_ci(k::Integer, n::Integer; z::Real = 1.96)
    n == 0 && return (0.0, 1.0)
    p = k / n
    z2 = z^2
    denom = 1 + z2 / n
    center = (p + z2 / (2n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z2 / (4n^2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))
end

"""
    bootstrap_ci(metric_fn, y_true, y_pred; n_boot=2000, alpha=0.05, rng) -> (point, lo, hi)

Percentile bootstrap CI for any trial-level metric. `metric_fn` must accept two
Boolean vectors `(y_true, y_pred)` and return a scalar. Trials are resampled with
replacement, preserving the (true, pred) pairing. Returns the point estimate on
the full data plus the (1-alpha) interval.
"""
function bootstrap_ci(metric_fn, y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool};
                      n_boot::Int = 2000, alpha::Real = 0.05, rng = Random.GLOBAL_RNG)
    n = length(y_true)
    point = metric_fn(y_true, y_pred)
    n == 0 && return (point, NaN, NaN)
    stats = Vector{Float64}(undef, n_boot)
    idx = Vector{Int}(undef, n)
    @inbounds for b in 1:n_boot
        for i in 1:n
            idx[i] = rand(rng, 1:n)
        end
        stats[b] = metric_fn(view(y_true, idx), view(y_pred, idx))
    end
    sort!(stats)
    lo = quantile_sorted(stats, alpha / 2)
    hi = quantile_sorted(stats, 1 - alpha / 2)
    return (point, lo, hi)
end

# Linear-interpolated quantile of an already-sorted vector (avoids a Statistics dep quirk).
function quantile_sorted(sorted::AbstractVector{<:Real}, q::Real)
    n = length(sorted)
    n == 0 && return NaN
    n == 1 && return float(sorted[1])
    h = (n - 1) * q + 1
    lo = floor(Int, h)
    hi = ceil(Int, h)
    lo == hi && return float(sorted[lo])
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo])
end

"""
    auroc(scores, labels) -> Float64

Area under the ROC curve via the Mann–Whitney U statistic, with correct
average-rank handling of ties. `scores` are continuous anomaly scores (higher =
more anomalous), `labels` Boolean (`true` = positive/anomaly). Threshold-free:
0.5 = chance, 1.0 = perfect ranking.
"""
function auroc(scores::AbstractVector{<:Real}, labels::AbstractVector{Bool})
    length(scores) == length(labels) || throw(ArgumentError("length mismatch"))
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
    average_precision(scores, labels) -> Float64

Area under the precision–recall curve (average precision). More informative than
AUROC when the negative class dominates. Higher = better; the no-skill baseline
equals the positive-class prevalence.
"""
function average_precision(scores::AbstractVector{<:Real}, labels::AbstractVector{Bool})
    length(scores) == length(labels) || throw(ArgumentError("length mismatch"))
    n_pos = count(labels)
    n_pos == 0 && return NaN
    order = sortperm(scores; rev = true)   # most-anomalous first
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    @inbounds for k in eachindex(order)
        labels[order[k]] ? (tp += 1) : (fp += 1)
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    end
    return ap
end

"""
    summarize(y_true, y_pred; scores=nothing, n_boot=2000, alpha=0.05, rng) -> NamedTuple

One-stop computation of every metric plus bootstrap CIs for the imbalance-aware
headline numbers. If `scores` is supplied, AUROC and average precision are added.
"""
function summarize(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool};
                   scores::Union{Nothing,AbstractVector{<:Real}} = nothing,
                   n_boot::Int = 2000, alpha::Real = 0.05, rng = Random.GLOBAL_RNG)
    c = confusion_counts(y_true, y_pred)
    pcr = per_class_precision_recall(c)

    ba_ci  = bootstrap_ci((yt, yp) -> balanced_acc(confusion_counts(collect(yt), collect(yp))),
                          y_true, y_pred; n_boot = n_boot, alpha = alpha, rng = rng)
    mcc_ci = bootstrap_ci((yt, yp) -> mcc(confusion_counts(collect(yt), collect(yp))),
                          y_true, y_pred; n_boot = n_boot, alpha = alpha, rng = rng)
    f1_ci  = bootstrap_ci((yt, yp) -> f1_score(confusion_counts(collect(yt), collect(yp))),
                          y_true, y_pred; n_boot = n_boot, alpha = alpha, rng = rng)

    sens_ci = wilson_ci(c.tp, c.tp + c.fn)
    spec_ci = wilson_ci(c.tn, c.tn + c.fp)

    auc = scores === nothing ? nothing : auroc(scores, y_true)
    ap  = scores === nothing ? nothing : average_precision(scores, y_true)

    return (
        confusion          = c,
        matrix             = [c.tp c.fn; c.fp c.tn],
        accuracy           = accuracy(c),
        balanced_accuracy  = balanced_acc(c),
        balanced_acc_ci    = (ba_ci[2], ba_ci[3]),
        mcc                = mcc(c),
        mcc_ci             = (mcc_ci[2], mcc_ci[3]),
        precision_anomaly  = pcr.anomaly.precision,
        recall_anomaly     = pcr.anomaly.recall,
        precision_healthy  = pcr.healthy.precision,
        recall_healthy     = pcr.healthy.recall,
        per_class          = pcr,
        f1                 = f1_score(c),
        f1_ci              = (f1_ci[2], f1_ci[3]),
        f2                 = fbeta_score(c; beta = 2.0),
        gmean              = gmean(c),
        youden_j           = youden_j(c),
        cohens_kappa       = cohens_kappa(c),
        sensitivity        = recall_score(c),
        sensitivity_ci     = sens_ci,
        specificity        = specificity(c),
        specificity_ci     = spec_ci,
        npv                = npv(c),
        auroc              = auc,
        average_precision  = ap,
    )
end

end # module ClassificationMetrics
