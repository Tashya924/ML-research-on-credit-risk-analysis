# Full credit-card generation comparison

All methods use the original CSV and matching real test splits. Scores below use validation-selected F1 cutoffs. AP shows mean ± split standard deviation; other scores show means.

| Method | Model | AP | AUC | F1 | Precision | Recall | Accuracy | Balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Diffusion with MLP | Logistic regression | 0.4692 ± 0.0123 | 0.7190 | 0.4866 | 0.4662 | 0.5125 | 0.7618 | 0.6724 |
| Diffusion with MLP | Histogram gradient boosting | 0.5439 ± 0.0123 | 0.7818 | 0.5419 | 0.5073 | 0.5828 | 0.7827 | 0.7111 |
| Diffusion with MLP | Random forest | 0.5459 ± 0.0122 | 0.7747 | 0.5313 | 0.5086 | 0.5580 | 0.7826 | 0.7021 |
| Diffusion with MLP | Extra trees | 0.5374 ± 0.0127 | 0.7679 | 0.5232 | 0.5178 | 0.5328 | 0.7858 | 0.6951 |
| Diffusion with Transformer | Logistic regression | 0.5222 ± 0.0082 | 0.7567 | 0.5214 | 0.4928 | 0.5565 | 0.7746 | 0.6964 |
| Diffusion with Transformer | Histogram gradient boosting | 0.5276 ± 0.0079 | 0.7682 | 0.5218 | 0.5108 | 0.5376 | 0.7831 | 0.6949 |
| Diffusion with Transformer | Random forest | 0.5336 ± 0.0116 | 0.7652 | 0.5194 | 0.5020 | 0.5421 | 0.7792 | 0.6942 |
| Diffusion with Transformer | Extra trees | 0.5290 ± 0.0068 | 0.7600 | 0.5176 | 0.4659 | 0.5828 | 0.7605 | 0.6967 |
| Original training data | Logistic regression | 0.5389 ± 0.0097 | 0.7668 | 0.5330 | 0.5139 | 0.5545 | 0.7857 | 0.7028 |
| Original training data | Histogram gradient boosting | 0.5533 ± 0.0105 | 0.7805 | 0.5415 | 0.5223 | 0.5642 | 0.7897 | 0.7088 |
| Original training data | Random forest | 0.5511 ± 0.0090 | 0.7773 | 0.5413 | 0.5246 | 0.5608 | 0.7903 | 0.7081 |
| Original training data | Extra trees | 0.5453 ± 0.0102 | 0.7717 | 0.5309 | 0.5262 | 0.5389 | 0.7898 | 0.6999 |
| AGSS + diffusion with MLP | Logistic regression | 0.4303 ± 0.0430 | 0.6955 | 0.4633 | 0.4282 | 0.5160 | 0.7350 | 0.6565 |
| AGSS + diffusion with MLP | Histogram gradient boosting | 0.5389 ± 0.0075 | 0.7794 | 0.5379 | 0.5124 | 0.5666 | 0.7851 | 0.7068 |
| AGSS + diffusion with MLP | Random forest | 0.5437 ± 0.0101 | 0.7720 | 0.5305 | 0.4992 | 0.5710 | 0.7768 | 0.7030 |
| AGSS + diffusion with MLP | Extra trees | 0.5332 ± 0.0112 | 0.7625 | 0.5197 | 0.4866 | 0.5623 | 0.7704 | 0.6958 |
| AGSS + diffusion with Transformer | Logistic regression | 0.5252 ± 0.0106 | 0.7559 | 0.5231 | 0.5132 | 0.5389 | 0.7830 | 0.6956 |
| AGSS + diffusion with Transformer | Histogram gradient boosting | 0.5272 ± 0.0051 | 0.7631 | 0.5244 | 0.4883 | 0.5681 | 0.7727 | 0.6994 |
| AGSS + diffusion with Transformer | Random forest | 0.5323 ± 0.0096 | 0.7652 | 0.5202 | 0.5103 | 0.5331 | 0.7828 | 0.6932 |
| AGSS + diffusion with Transformer | Extra trees | 0.5318 ± 0.0021 | 0.7590 | 0.5167 | 0.4800 | 0.5628 | 0.7673 | 0.6940 |
| AGSS interpolation (DBSCAN-guided) | Logistic regression | 0.5361 ± 0.0071 | 0.7648 | 0.5315 | 0.5064 | 0.5616 | 0.7817 | 0.7029 |
| AGSS interpolation (DBSCAN-guided) | Histogram gradient boosting | 0.5102 ± 0.0169 | 0.7526 | 0.5075 | 0.4519 | 0.5809 | 0.7511 | 0.6900 |
| AGSS interpolation (DBSCAN-guided) | Random forest | 0.5224 ± 0.0103 | 0.7610 | 0.5180 | 0.4865 | 0.5566 | 0.7713 | 0.6944 |
| AGSS interpolation (DBSCAN-guided) | Extra trees | 0.5336 ± 0.0099 | 0.7629 | 0.5235 | 0.5201 | 0.5280 | 0.7882 | 0.6949 |

For individual splits and cutoff 0.5, use `results/credit_card/metrics.csv`.
