# QuadFDB

**QuadFDB: Benchmarking Quadrotor Motor Fault Diagnosis with Real Flight Data**

Junhao Hou · Meng Zhao · Qiang Shen* · Jing Wang · Xiaohui Zhang<br>
School of Aeronautics and Astronautics, Shanghai Jiao Tong University<br>
*Corresponding author

[Project page](https://markjh2001.github.io/QuadFDB/)

QuadFDB (Quadrotor Fault Diagnosis Benchmark) studies detection and motor isolation during short, controlled reductions of one quadrotor motor's PWM command. The study contains 1,560 fault flights and 120 healthy controls, covering four motors, three temporal profiles, thirteen target multipliers, and ten repetitions per fault condition. These are command-level interventions under real sensing and feedback, rather than measurements of physical motor or propeller damage.

This private research repository covers this paper. The broader CrazyFDD research project and its other experiments are maintained separately. Raw flight recordings, prepared flight arrays, fitted checkpoints, and manuscript source/PDF files are **not included**.

## Code and results

| Component | Included implementation |
| --- | --- |
| RAGB | Healthy linear motion prediction, signed residual sequences, local RF class probabilities, and histogram gradient boosting |
| Controlled comparisons | A shared 201-column feature bank, four input masks, HGB and five-class CE MLP heads |
| Tabular references | Local RF, flattened-history Extra Trees, RBF SVM, and raw-window HGB |
| Neural references | MLP, FCN, LSTM, TCN, and Transformer architectures; four-output BCE training and inference helpers |
| Evaluation | Healthy-CAL q99 threshold, fresh two-endpoint alarms, first-alarm motor isolation, false alarms, coverage, censoring, and conditional delay |
| Results | Paper aggregates, per-seed event summaries, and the fixed head/input attribution study; no endpoint scores or flight arrays |

The package extracts the numerical dependency closure from the study implementation. It does not require a VLM, image renderer, local research checkout, flight controller, or server scheduler. Source hashes and extraction changes are recorded in [protocol/source_manifest.json](protocol/source_manifest.json). The portable checkpoint format differs from the internal server format. New checkpoints use the `quadfdb` namespace; older `quadfdi` joblib snapshots require the original package or explicit migration.

## RAGB

Residual-Augmented Gradient Boosting (RAGB) augments the input representation of a standard histogram gradient boosting classifier. The boosting loss and update rule are unchanged.

Each causal window contains sixteen samples over 300 ms: takeoff-relative position, body angular rates, and the four motor commands before injection. A healthy ridge predictor estimates current position and angular rates from the preceding ten motion/command observations. Six signed prediction-error vectors cover the final 100 ms. A local RF uses the latest eleven physical observations plus endpoint, mean, change, RMS, and slope summaries to produce five class-probability estimates.

The final representation contains **160 normalized raw values + 36 residual values + 5 RF probabilities**. The response and RF models exclude each target flight when constructing fitting features. Residual centering and scaling use pooled healthy errors from these held-out flights; raw scaling uses all healthy FIT observations. Calibration and test features use tools fitted on the full permitted FIT set. All stochastic components share the pipeline seed.

The four controlled masks preserve the 201 input slots and zero excluded components during both fitting and inference. The categorical control MLP has five CE outputs. It is distinct from the main comparison's four-output BCE MLP, so their raw-input results are not interchangeable.

## Installation and checks

```bash
python -m pip install -e .
# Optional: neural architectures and fitting helpers
python -m pip install -e '.[neural]'
python -m unittest discover -s tests -v
```

The paper's sklearn runs used version 1.9.1; the dependency is pinned accordingly. Neural study runs used PyTorch 2.9.1 with CUDA. CPU smoke checks validate the extracted computations and interfaces; they do not claim bitwise reproduction of the original CUDA training or reproduce the paper's flight results.

## Working with separately held data

Training requires a **prepared FIT-only window archive** supplied outside this repository. The recording decoder, synchronization/export pipeline, and private split membership are not bundled. Merely installing the package does not reproduce the dataset.

The example training command accepts an NPZ archive with:

| Key | Shape / meaning |
| --- | --- |
| `windows` | `N × 16 × 10` finite physical observations; channel order is `quadfdb.FEATURE_NAMES` |
| `motor_health` | `N × 4` binary labels: 1 healthy, 0 faulty; at most one faulty motor |
| `flight_ids` | `N` integer or Unicode identifiers shared by all windows from a physical flight |
| `roles` | `N` Unicode entries, all equal to `fit` |

The caller must select windows using the fixed flight partition and eligibility rules in [protocol/benchmark.json](protocol/benchmark.json): retain eligible fault endpoints, cap eligible healthy windows at 64 per flight, and require a healthy fitting history to be healthy throughout. Injection parameters, applied PWM, fault timing, and labels never become inference inputs. The role check validates the supplied declaration; it cannot independently authenticate private split membership.

```bash
python scripts/fit_ragb.py \
  --fit-npz /path/to/prepared_fit.npz \
  --output checkpoints/mild/20261009 \
  --seed 20261009
```

Inference and evaluation use physical observation windows and separately prepared eligible calibration scores / scored episode records:

```python
from quadfdb import RAGB, calibrate_q99, evaluate

model = RAGB.load("checkpoints/mild/20261009")
scores = model.score(physical_windows)  # N × 16 × 10; no event metadata
threshold = calibrate_q99(eligible_healthy_cal_scores)
report = evaluate(scored_test_episodes, threshold)
```

`scored_test_episodes` must preserve timestamps, input/score validity, healthy exposure, event bounds, and motor labels in the evaluator's documented schema. A simple array of window predictions is insufficient to reconstruct event metrics. Thresholds come from healthy CAL only, not TEST. Training/evaluation access and role selection remain the responsibility of the caller.

## Evaluation scope

The paper uses two fixed training tasks: Mild (288 FIT / 96 CAL / 96 TEST flights) and pooled Moderate/Severe (792 / 264 / 264). The tasks share the same 120 healthy controls and contain 1,680 unique physical flights. Moderate and Severe use the same pooled model and threshold, with 96 and 144 faulty test flights; Mild has 72. Each group includes the same 24 healthy test controls. Three pipeline seeds retain the same physical partitions.

The alarm rule uses the higher-order empirical healthy-CAL q99, a strict score exceedance, and two adjacent valid 20 ms endpoints. A fresh run must start at or after onset and confirm before the actual intervention ends. Joint isolation requires the correct unique motor at the first alarm. False alarms use pooled eligible healthy exposure. Delay is conditional on detection, includes wrong-motor alarms, and excludes processing and communication costs.

Earlier held-out results informed method development, so the study is a **retrospective evaluation**, not a new blind external test. Three seed repeats do not establish statistical significance or cross-platform generalization. PWM reductions and hovering flights do not establish performance on physical damage or maneuvers.

The archived [benchmark results](results/benchmark.csv) and [controlled comparisons](results/controlled_ablation.json) retain all specified outcomes. RAGB's Mild isolation advantage over the local RF comes with more false alarms; full-input fusion is not best in every severity group. See [results/README.md](results/README.md) for the statistical and provenance boundaries.
