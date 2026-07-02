#!/usr/bin/env bash
# run_all.sh - set up the Julia environment and regenerate the project's
# figures, demo animation, and (optionally) the headline classification
# result.
#
# Usage:
#   ./run_all.sh setup     # instantiate the Julia project (run this first)
#   ./run_all.sh figures   # regenerate the fast, single-recording figures
#   ./run_all.sh demo      # regenerate the healthy-vs-infarction demo GIF
#   ./run_all.sh results   # regenerate the classifier figures (SLOW, see below)
#   ./run_all.sh confirm   # reproduce the headline AUROC result (SLOW)
#   ./run_all.sh all       # setup + figures + demo (does NOT include the slow steps)
#
# "SLOW" steps run the full STDP simulation over dozens of ECG recordings
# per class and can take from several minutes to about an hour on a laptop,
# since figures/cache/ (the memoized feature cache) is not checked into git.
# Run them once and let them finish; subsequent figure-only reruns reuse the
# cache.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

JULIA="${JULIA:-julia}"
PROJECT_FLAG=(--project=.)

setup() {
    echo "==> Instantiating Julia project (this installs all packages; first run may take a while)"
    "$JULIA" "${PROJECT_FLAG[@]}" -e 'using Pkg; Pkg.instantiate()'
}

figures() {
    echo "==> Regenerating fast figures (figures/*.png and step*.pdf at repo root)"
    for f in figures/plot_pipeline.jl figures/plot_lif_trace.jl figures/plot_stdp_window.jl \
             figures/plot_delta_modulation.jl figures/plot_pqrst_annotated.jl \
             figures/plot_healthy_vs_infarction.jl; do
        echo "  -- $f"
        "$JULIA" "${PROJECT_FLAG[@]}" "$f"
    done
}

demo() {
    echo "==> Regenerating the healthy-vs-infarction demo GIF (docs/imgs/demo_healthy_vs_infarction.gif)"
    "$JULIA" "${PROJECT_FLAG[@]}" examples/demo_healthy_vs_infarction.jl
}

results() {
    echo "==> Regenerating classifier figures (confusion matrix, ROC, feature scatter)"
    echo "    NOTE: this runs the full routed multi-lead network over many recordings."
    echo "    Expect this to take a while on first run; results are cached in figures/cache/."
    for f in figures/plot_confusion_matrix.jl figures/plot_roc_heldout.jl figures/plot_feature_scatter.jl; do
        echo "  -- $f"
        "$JULIA" "${PROJECT_FLAG[@]}" "$f"
    done
}

confirm() {
    echo "==> Reproducing the headline test-set AUROC result (confirm_routed_multilead.jl)"
    echo "    NOTE: this is the slowest step — multiple seeds x lead sets, each a full sweep."
    "$JULIA" "${PROJECT_FLAG[@]}" confirm_routed_multilead.jl
}

case "${1:-all}" in
    setup)   setup ;;
    figures) figures ;;
    demo)    demo ;;
    results) results ;;
    confirm) confirm ;;
    all)     setup; figures; demo ;;
    *)
        echo "Unknown target: ${1:-}" >&2
        echo "Usage: $0 {setup|figures|demo|results|confirm|all}" >&2
        exit 1
        ;;
esac
