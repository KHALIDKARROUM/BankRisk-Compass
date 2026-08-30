# Aegis-Credit Final Report

## Final Model

The final model is a calibrated, leakage-safe Gradient Boosting classifier. Missing values, scaling, and one-hot encoding are fitted only on training data. Model selection, probability calibration, and threshold selection use separate data partitions; the final metrics below are measured once on an untouched test set.

Application-time scoring intentionally excludes lender-assigned fields (`loan_grade` and `loan_int_rate`) to avoid using information that may not exist when an applicant is first assessed.
Age is also excluded from the probability model; it is retained only for input plausibility checks and subgroup monitoring.

Data split:

- Training: 3,600 rows
- Model selection: 480 rows
- Probability calibration: 360 rows
- Threshold selection: 360 rows
- Final test: 1,200 rows

Selected model parameters:

```text
{'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 1}
```

## Default 0.50 Threshold Results

| Metric | Score |
|---|---:|
| Accuracy | 0.844 |
| Precision | 0.500 |
| Recall | 0.043 |
| F1-score | 0.079 |
| ROC-AUC | 0.691 |
| Average precision | 0.312 |
| Brier score | 0.122 |

## Business Threshold Results

The selected screening threshold is **0.15**. Its illustrative count-weighted objective assumes a false negative is 5x more costly than a false positive; it is not an approved expected-loss model:

- False negative: a risky borrower is approved.
- False positive: a safer borrower is unnecessarily routed to manual review.

| Metric | Score |
|---|---:|
| Accuracy | 0.653 |
| Precision | 0.248 |
| Recall | 0.604 |
| F1-score | 0.352 |
| ROC-AUC | 0.691 |
| Average precision | 0.312 |
| Brier score | 0.122 |
| False positives | 342 |
| False negatives | 74 |
| Business cost | 712 |

## Interpretation

The model is a decision-support tool, not an autonomous approval system. The business threshold prioritizes recall for defaults while monitoring the number of safer applicants routed to review. Age-group diagnostics and calibration reports are generated for governance review, but they do not replace a complete fair-lending assessment using legally appropriate protected-class data.
