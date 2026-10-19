# AGENTS.md — FreshCheck

This document defines the operational contract for AI coding agents and contributors working on FreshCheck.

The goal is to prevent scope drift, incorrect class mapping, leakage, broken preprocessing, inflated metrics, unsupported food-safety claims, destructive Git operations, and loss of experiment evidence.

---

## 1. Project Identity

```text
Project          FreshCheck
Portfolio        12%
Task             Visual produce-quality classification
Model            EfficientNet-B0
Classes          5
App              Streamlit
```

Champion checkpoint:

```text
models/freshcheck_effnet_b0_best.pt
```

Class mapping:

```text
0 Damaged
1 Dried
2 Old
3 Ripe
4 Unripe
```

Crop mapping:

```text
0 Bell Pepper
1 Chile Pepper
2 New Mexico Green Chile
3 Tomato
```

---

## 2. Source of Truth

Use this precedence:

```text
1. reports/final_summary.json
2. reports/test_classification_report.csv
3. reports/test_per_crop_metrics.csv
4. reports/dataset_audit.json
5. reports/split_manifest.csv
6. reports/training_history.csv
7. reports/inference_config.json
8. notebooks/freshcheck_fruit.ipynb
9. README.md
```

Never rewrite machine-readable reports simply to make prose agree with them.

---

## 3. Locked Champion

```text
Model      EfficientNet-B0
Best epoch 8
```

Held-out test:

```text
Accuracy          99.46%
Balanced Accuracy 99.64%
Macro F1          99.57%
```

Weakest class F1:

```text
≈ 99.25%
```

Per-crop:

```text
Bell Pepper            99.44%
Chile Pepper           100.00%
New Mexico Green Chile 97.98%
Tomato                 100.00%
```

Do not replace these values without a new documented evaluation.

---

## 4. Claim Boundaries

Supported:

```text
visual produce-quality classification
five-class classification
EfficientNet-B0
mounted VegNet internal benchmark
local Streamlit MVP
```

Unsupported:

```text
food safety
microbial contamination
chemical freshness
shelf-life prediction
edible/not-edible
universal fruit quality
universal vegetable quality
commercial certification
```

Never turn `99.46% held-out accuracy` into `99.46% real-world freshness accuracy`.

---

## 5. Dataset Contract

Operational dataset family:

```text
Project-AgML/VegNet_quality_classification
```

Mounted parquet:

```text
/kaggle/input/datasets/loliwibu/vegnet/data/train-00000-of-00001.parquet
```

Mounted rows:

```text
6150
```

Columns:

```text
image
label
crop_type
```

---

## 6. Class Mapping Contract

Required order:

```text
0 = Damaged
1 = Dried
2 = Old
3 = Ripe
4 = Unripe
```

Agents must never reorder classes without explicit remapping/retraining, evaluation, inference-config update, and documentation.

A wrong class order can make correct logits display wrong labels.

---

## 7. Crop Mapping Contract

Required:

```text
0 = Bell Pepper
1 = Chile Pepper
2 = New Mexico Green Chile
3 = Tomato
```

Never claim validation on crop types absent from the mounted data.

---

## 8. Dataset Distribution

```text
Damaged = 317
Dried   = 1389
Old     = 2044
Ripe    = 1787
Unripe  = 613
```

Crop counts:

```text
Bell Pepper            = 1176
Chile Pepper           = 1286
New Mexico Green Chile = 1327
Tomato                 = 2361
```

Total:

```text
6150
```

---

## 9. Dataset Audit Contract

Expected:

```text
Readable images          = 6150
Unreadable images        = 0
Exact duplicates         = 0
Label conflicts          = 0
Cross-split hash leakage = 0
```

Do not silently weaken these checks.

---

## 10. Split Contract

```text
train = 4305
val   = 922
test  = 923
```

Never:

- train on test;
- tune on test;
- select champion on test;
- move samples silently;
- alter preprocessing based on test failures.

Any split revision must be explicitly versioned.

---

## 11. Training Contract

Training notebook:

```text
notebooks/freshcheck_fruit.ipynb
```

Training should record:

```text
seed
split
architecture
input size
normalization
optimizer
learning rate
epochs
batch size
scheduler
checkpoint rule
```

---

## 12. Model Promotion Checklist

A replacement champion requires:

```text
[ ] dataset version recorded
[ ] class/crop mappings verified
[ ] split integrity verified
[ ] no leakage
[ ] validation complete
[ ] held-out test complete
[ ] per-class metrics reviewed
[ ] per-crop metrics reviewed
[ ] failure cases reviewed
[ ] inference config updated
[ ] app smoke-tested
[ ] reports written
[ ] README updated
```

---

## 13. Evaluation Metrics

Primary metrics:

```text
Accuracy
Balanced Accuracy
Macro F1
Per-Class Precision
Per-Class Recall
Per-Class F1
Per-Crop Metrics
Confusion Matrix
```

Do not rely on accuracy alone when discussing imbalanced classes.

---

## 14. Inference Configuration

Source:

```text
reports/inference_config.json
```

Training and local inference must agree on:

```text
class_names
image_size
mean
std
architecture
```

Changing preprocessing is a model-interface change and requires retesting.

---

## 15. Inference Contract

Reusable module:

```text
src/inference.py
```

Result contract:

```text
predicted_class
predicted_index
confidence
probabilities
top3
model_name
image_size
source_name
saved_image
saved_json
```

Top-1 must equal the max-probability index.

Top-3 must be descending.

---

## 16. App Contract

```text
app/app.py
```

Framework:

```text
Streamlit
```

Expected UX:

- upload image;
- show input;
- show visual-quality prediction;
- show confidence;
- show probability chart;
- show Top-3;
- show saved result path;
- show prediction history;
- show interpretation boundary.

Preferred wording:

```text
Visual Quality
Prediction
Confidence
```

Avoid:

```text
Food Safety
Safe to Eat
Shelf Life
Freshness Guarantee
```

---

## 17. Saved Results

Generated local outputs:

```text
tests/results/
```

Typical:

```text
*_prediction.jpg
*_result.json
prediction_history.csv
```

Generated results are temporary unless intentionally copied into `assets/screenshots/` or `assets/predictions/`.

---

## 18. Asset Directory Contract

Dataset visuals:

```text
assets/dataset/
├── crop_distribution.png
├── crop_quality_distribution_heatmap.png
├── dataset_quality_examples.png
└── quality_class_distribution.png
```

Evaluation:

```text
assets/evaluation/
├── test_confusion_matrix.png
├── test_confusion_matrix_normalized.png
└── test_macro_f1_by_crop.png
```

Predictions:

```text
assets/predictions/
├── test_correct_predictions.png
└── test_failure_cases.png
```

Training:

```text
assets/training/
├── training_loss.png
└── training_macro_f1.png
```

Do not leave generated figures loose at `assets/` root.

---

## 19. Reports Contract

Reports include:

```text
dataset_audit.json
final_summary.json
inference_config.json
split_manifest.csv
test_classification_report.csv
test_per_crop_metrics.csv
test_predictions.csv
training_history.csv
```

Reports are evidence, not disposable output.

---

## 20. Reproducibility Policy

Every future experiment should record:

```text
date
dataset version
sample count
class mapping
crop mapping
split
seed
preprocessing
architecture
optimizer
learning rate
epochs
checkpoint rule
final metrics
artifact paths
```

Never overwrite the only copy of a previous experiment.

---

## 21. Testing Policy

Target coverage:

```text
checkpoint existence
checkpoint load
class count
class order
config parsing
valid inference
invalid image
probability sum
top1 correctness
top3 ordering
CPU inference
result image saving
result JSON saving
history append
app import
```

Unit tests must not require the full dataset.

---

## 22. Domain-Shift Rule

External images may differ in:

- background;
- lighting;
- camera;
- variety;
- framing;
- object scale;
- compression.

One wrong web-image prediction does not invalidate the internal benchmark.

Strong internal benchmark performance does not prove external reliability.

Both can be true.

---

## 23. Confidence Rule

Softmax confidence is not:

- food-safety probability;
- universal correctness probability;
- out-of-distribution confidence;
- calibration guarantee.

A classifier can be confidently wrong.

---

## 24. Food-Safety Rule

FreshCheck is not a medical, laboratory, chemical, or food-safety system.

Never generate UI/docs claiming:

```text
Safe to eat
Unsafe
Healthy
Contaminated
Shelf life: X days
```

without a separately validated task.

---

## 25. Deployment Policy

Current supported claim:

```text
Local Streamlit MVP
```

Future deployment must benchmark:

```text
CPU latency
GPU latency
RAM
VRAM
startup time
concurrency
input limits
failure behavior
```

Do not describe local Streamlit as production deployment.

---

## 26. Dependency Policy

Primary dependencies:

```text
torch
torchvision
streamlit
numpy
pandas
Pillow
```

Update `requirements.txt` whenever runtime requirements change.

Avoid unnecessary heavy packages.

---

## 27. Security and Privacy

Never commit:

```text
API keys
tokens
passwords
cloud credentials
.env secrets
```

Current inference is local.

Do not add silent uploads, analytics, tracking, or cloud persistence without explicit approval.

---

## 28. Git Policy

Use meaningful commits:

```text
feat:
fix:
docs:
test:
refactor:
chore:
```

No fake streak commits.

No blind force-push.

Before destructive cleanup:

```powershell
git status
git clean -fdn
```

Always dry-run first.

Never blindly run:

```powershell
git clean -fd
```

Untracked notebooks/reports/screenshots may be unrecoverable.

---

## 29. Safe Agent Changes

Agents may:

- improve code clarity;
- add type hints;
- add tests;
- improve app layout;
- improve errors;
- add batch inference;
- add export/download features;
- add deployment configuration;
- improve docs.

---

## 30. Changes Requiring Explicit Approval

Ask first before:

- retraining;
- replacing checkpoint;
- changing class mapping;
- changing crop mapping;
- changing split;
- changing preprocessing;
- deleting reports;
- deleting model weights;
- deleting notebooks;
- changing license;
- changing final benchmark claims;
- changing task scope.

---

## 31. Future Classes

Adding labels such as `Moldy`, `Fresh`, `Rotten`, or `Bruised` is a schema change.

Require:

```text
new label definition
new data audit
new model/evaluation
new inference config
new reports
new documentation
```

Do not silently map new concepts onto current labels.

---

## 32. Future Crop Expansion

Adding new produce types requires:

```text
data
label audit
split audit
evaluation
per-crop metrics
external testing
```

The fact that the app accepts an arbitrary image does not prove coverage.

---

## 33. Definition of Done

For ML changes:

```text
[ ] data version recorded
[ ] class mapping verified
[ ] crop mapping verified
[ ] leakage audit complete
[ ] validation complete
[ ] held-out test complete
[ ] per-class metrics reviewed
[ ] per-crop metrics reviewed
[ ] failures inspected
[ ] reports written
[ ] inference config updated
[ ] local inference tested
[ ] README updated
```

For app changes:

```text
[ ] app starts
[ ] checkpoint loads
[ ] upload works
[ ] prediction works
[ ] probabilities render
[ ] Top-3 renders
[ ] saved result works
[ ] wording respects claim boundaries
```

---

## 34. Before → Now → Next

### Before

```text
public dataset
no audit
no leakage check
no champion
no app
```

### Now

```text
audited mounted dataset
fixed class mapping
EfficientNet-B0 champion
held-out benchmark
per-crop analysis
failure review
Streamlit MVP
saved inference artifacts
```

### Next

```text
automated tests
external evaluation
confidence calibration
app polish
batch inference
ONNX
CPU benchmark
deployment
more crop species
```

---

## Final Agent Rule

FreshCheck answers:

> Which dataset-defined visual-quality category most closely matches this produce image?

FreshCheck does **not** answer:

> Is this food safe to eat, how many days of shelf life remain, or is this chemically fresh?

Every future code, UI, metric, and documentation change must preserve that distinction.
