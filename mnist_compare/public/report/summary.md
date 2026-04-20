# Public MNIST Compare (RNN vs SNN)

## Per-Run Metrics

| run | model | seed | epochs | loss_start | loss_end | loss_min | tf_mse | free_run_mse | free_run_pixel_mse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rnn_pred_seed0_e3000 | rnn | 0 | 3000 | 1837.98 | 294.942 | 43.9553 | 0.00912383 | 0.0108313 | 0.0559203 |
| rnn_pred_seed1_e3000 | rnn | 1 | 3000 | 1717.11 | 65.59 | 65.59 | 0.00236689 | 0.00821907 | 0.0393216 |
| rnn_pred_seed2_e3000 | rnn | 2 | 3000 | 2155.84 | 46.3094 | 43.534 | 0.00188122 | 0.00813437 | 0.0388031 |
| snn_pred_seed0_e3000 | snn | 0 | 3000 | 32133.3 | 94.5171 | 93.7591 | 0.00301346 | 0.00330042 | 0.0173994 |
| snn_pred_seed1_e3000 | snn | 1 | 3000 | 31855.7 | 108.921 | 106.177 | 0.00342106 | 0.00409341 | 0.0209773 |
| snn_pred_seed2_e3000 | snn | 2 | 3000 | 31968.6 | 92.4197 | 89.6065 | 0.002846 | 0.00355636 | 0.0183557 |

## Model-Wise Mean ± Std

| model | metric | mean | std | n |
|---|---|---:|---:|---:|
| rnn | loss_end | 135.614 | 138.319 | 3 |
| rnn | tf_mse | 0.00445731 | 0.00404861 | 3 |
| rnn | free_run_mse | 0.00906159 | 0.00153322 | 3 |
| rnn | free_run_pixel_mse | 0.0446817 | 0.00973638 | 3 |
| snn | loss_end | 98.6191 | 8.98272 | 3 |
| snn | tf_mse | 0.00309351 | 0.000295768 | 3 |
| snn | free_run_mse | 0.00365007 | 0.000404711 | 3 |
| snn | free_run_pixel_mse | 0.0189108 | 0.00185243 | 3 |

Artifacts:
- per_run_metrics.csv
- summary_by_model.csv
- compare_barplots.png