# Archived paper results

- `benchmark.csv`: 10 main comparison configurations × 3 severity groups.
- `benchmark.json`: unrounded means/sample SDs, 90 per-seed event records, and source-artifact hash identifiers.
- `controlled_ablation.json`: the fixed 54-cell study: 2 heads × 4 input masks × 2 tasks × 3 pipeline seeds, plus 6 frozen local-RF references. Contains aggregate and paired-count summaries, not endpoint scores or individual flight arrays.

The main table's RF is the local 200 ms forest that also supplies the posterior input. It is not the broader project's earlier flattened-window RF. RAGB is the complete raw/residual/posterior HGB configuration. The main MLP uses four BCE outputs; the controlled MLP uses five-class CE with a fixed 201-input architecture.

Means and sample SDs (`ddof=1`) summarize three training seeds on fixed test flights. SVM has three genuine fits per task, with identical outcomes under its deterministic `probability=False` recipe. Moderate and Severe share the same pooled training task. Their score-derived point metrics belong to that pooled task, not separately invented severity subsets.

Delay is the mean of per-run conditional detection medians. The paired-delay auxiliary compares the same detected physical flights before reducing per-seed differences; it must not be replaced by a subtraction of aggregate delays. Non-detections, censoring, wrong first motors, and false-alarm exposure retain the native definitions.

Source SHA-256 identifiers link these exports to independently checked internal result artifacts. Those private artifacts and original split memberships are not included. Exporting aggregates does not reproduce or newly evaluate a model. The held-out results were involved in earlier development; these results should be described as retrospective.
