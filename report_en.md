# CIFAR-10 KNN Experiment Results

- Training samples: 5,000 / Separate test samples: 1,000
- Images: 32×32 RGB, 3072 features
- Sampling and splitting seed: 42
- Validation: The same stratified 5-fold splits were used for all combinations
- Precision, Recall, F1: Macro averages, giving equal weight to each class
- Standard deviation: Sample standard deviation of the 5 fold scores (ddof=1); this does not represent a confidence interval

## Cross-Validation Comparison

| Distance | k | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|---:|
| manhattan | 7 | 0.2880 ± 0.0086 | 0.3637 ± 0.0189 | 0.2880 ± 0.0086 | 0.2763 ± 0.0086 |
| manhattan | 9 | 0.2870 ± 0.0109 | 0.3607 ± 0.0151 | 0.2870 ± 0.0109 | 0.2729 ± 0.0102 |
| manhattan | 5 | 0.2860 ± 0.0081 | 0.3441 ± 0.0087 | 0.2860 ± 0.0081 | 0.2743 ± 0.0043 |
| manhattan | 1 | 0.2814 ± 0.0084 | 0.3170 ± 0.0101 | 0.2814 ± 0.0084 | 0.2779 ± 0.0078 |
| cosine | 1 | 0.2780 ± 0.0106 | 0.3092 ± 0.0105 | 0.2780 ± 0.0106 | 0.2686 ± 0.0108 |
| cosine | 9 | 0.2740 ± 0.0102 | 0.3404 ± 0.0125 | 0.2740 ± 0.0102 | 0.2498 ± 0.0124 |
| cosine | 7 | 0.2722 ± 0.0081 | 0.3370 ± 0.0105 | 0.2722 ± 0.0081 | 0.2506 ± 0.0103 |
| cosine | 5 | 0.2710 ± 0.0085 | 0.3279 ± 0.0085 | 0.2710 ± 0.0085 | 0.2519 ± 0.0077 |
| manhattan | 3 | 0.2706 ± 0.0130 | 0.3438 ± 0.0158 | 0.2706 ± 0.0130 | 0.2594 ± 0.0112 |
| euclidean | 7 | 0.2630 ± 0.0082 | 0.3455 ± 0.0082 | 0.2630 ± 0.0082 | 0.2400 ± 0.0045 |
| euclidean | 9 | 0.2622 ± 0.0152 | 0.3463 ± 0.0213 | 0.2622 ± 0.0152 | 0.2372 ± 0.0126 |
| cosine | 3 | 0.2602 ± 0.0154 | 0.3319 ± 0.0173 | 0.2602 ± 0.0154 | 0.2392 ± 0.0161 |
| euclidean | 5 | 0.2586 ± 0.0053 | 0.3275 ± 0.0050 | 0.2586 ± 0.0053 | 0.2400 ± 0.0056 |
| euclidean | 1 | 0.2564 ± 0.0134 | 0.2978 ± 0.0187 | 0.2564 ± 0.0134 | 0.2460 ± 0.0143 |
| euclidean | 3 | 0.2452 ± 0.0082 | 0.3301 ± 0.0165 | 0.2452 ± 0.0082 | 0.2255 ± 0.0116 |

## Separate Test Results for the Selected Model

**manhattan, k=7** was selected because it achieved the highest mean cross-validation Accuracy. Ties were resolved by higher mean macro F1, then smaller k, then alphabetical order of distance names. The test data were not used for model selection and were evaluated once after selection.

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---:|---:|---:|---:|
| 0.3010 | 0.3669 | 0.3010 | 0.2843 |

## Brief Comparison and Explanation

- euclidean: The best k was 7, with a mean Accuracy of 0.2630. Accuracy ranged from 0.2452 to 0.2630 across the tested k values.
- manhattan: The best k was 7, with a mean Accuracy of 0.2880. Accuracy ranged from 0.2706 to 0.2880 across the tested k values.
- cosine: The best k was 1, with a mean Accuracy of 0.2780. Accuracy ranged from 0.2602 to 0.2780 across the tested k values.
- The selected combination achieved a CV Accuracy of 0.2880 and a test Accuracy of 0.3010. These values may differ because they were calculated on different samples.
- With a small k, predictions are strongly influenced by a few nearby samples. A larger k can stabilize majority voting, but may also include samples from other classes. Therefore, performance does not necessarily increase monotonically with k.
- Euclidean distance squares pixel differences, giving greater influence to large differences, while Manhattan distance sums absolute differences. Cosine distance compares vector directions rather than magnitudes. The relative performance of these distances should be judged within the scope of the experiments above.
- Pixel-based KNN is sensitive to changes in object position, background, and lighting. No feature extractor that learns the semantic content of objects was used.
- These results are based on a stratified subset rather than the entire CIFAR-10 dataset. They should not be reported as representing performance on the full dataset.

## Generated Files

- `cv_folds.csv`: Scores for each combination and fold (the timing is the prediction time shared by all k values for the same fold and distance)
- `cv_summary.csv`: Cross-validation means and standard deviations
- `test_metrics.csv`, `test_per_class.csv`: Overall and per-class final test evaluations
- `test_confusion_matrix.csv`, `test_predictions.csv`: Confusion matrix and predictions
- `cv_comparison.png`, `test_confusion_matrix.png`: Result plots
- `config.json`: Experiment settings and the selected combination
