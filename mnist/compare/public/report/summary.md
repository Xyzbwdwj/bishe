# Public MNIST Compare (RNN vs SNN)

## Per-Run Metrics

| run | model | seed | epochs | loss_start | loss_end | loss_min | tf_mse | free_run_mse | free_run_pixel_mse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rnn_pred_seed0_e100_benchmark | rnn | 0 | 100 | 1837.98 | 264.906 | 264.906 | 0.00826181 | 0.00888599 | 0.0471308 |
| rnn_pred_seed0_e3000 | rnn | 0 | 3000 | 1837.98 | 294.942 | 43.9553 | 0.00912383 | 0.0108313 | 0.0559203 |
| snn_pred_seed0_e100_benchmark | snn | 0 | 100 | 3435.22 | 456.55 | 450.983 | 0.0143081 | 0.0198841 | 0.099654 |
| snn_pred_seed0_e3000 | snn | 0 | 3000 | 3435.22 | 323.081 | 316.71 | 0.00961791 | 0.0175557 | 0.0820319 |
| snn_pred_analog_seed1_e3000 | snn | 1 | 3000 | 4006.91 | 327.082 | 324.573 | 0.00973619 | 0.015249 | 0.0720735 |
| snn_pred_analog_seed1_e500 | snn | 1 | 500 | 4006.91 | 343.791 | 336.397 | 0.0103424 | 0.0176658 | 0.0837797 |
| snn_pred_onoff_seed1_e500 | snn | 1 | 500 | 3914.61 | 348.904 | 347.218 | 0.0105379 | 0.0168065 | 0.0817073 |
| snn_pred_signed_seed1_e500 | snn | 1 | 500 | 4250.5 | 350.608 | 349.755 | 0.010515 | 0.01402 | 0.0695952 |
| snn_pred_analog_h400_seed2_e500 | snn | 2 | 500 | 3782.83 | 345.289 | 325.696 | 0.0104986 | 0.0190507 | 0.0899253 |
| snn_pred_analog_lrstep_seed3_e3000 | snn | 3 | 3000 | 3659.59 | 347.44 | 328.94 | 0.0101965 | 0.0159437 | 0.0760155 |
| snn_pred_analog_thr05_ref0_seed4_e500 | snn | 4 | 500 | 5835.17 | 359.189 | 358.154 | 0.0110442 | 0.0270243 | 0.126378 |
| snn_pred_analog_thr075_ref0_seed4_e500 | snn | 4 | 500 | 5783.97 | 358.57 | 356.264 | 0.0110231 | 0.0282973 | 0.131795 |
| snn_pred_analog_a085_b095_seed5_e500 | snn | 5 | 500 | 3515.09 | 345.305 | 337.117 | 0.0104194 | 0.0164414 | 0.0794186 |
| snn_pred_analog_ab095_seed5_e500 | snn | 5 | 500 | 3702.43 | 336.344 | 336.344 | 0.0101721 | 0.0150728 | 0.0724978 |
| snn_pred_analog_lrstep_finetune_seed6_e2000 | snn | 6 | 2000 | 400.76 | 357.215 | 344.057 | 0.0106662 | 0.016424 | 0.0792101 |
| snn_pred_analog_lrstep_plain_ft_seed7_e1000 | snn | 7 | 1000 | 341.665 | 346.52 | 329.335 | 0.0103403 | 0.0161461 | 0.0770526 |
| snn_pred_analog_ns20_seed8_e100_benchmark | snn | 8 | 100 | 15267.4 | 1698.46 | 1666.06 | 0.0126899 | 0.0190373 | 0.0941544 |
| snn_pred_analog_ns20_seed8_e1500 | snn | 8 | 1500 | 15267.4 | 1411.7 | 1381.23 | 0.0105672 | 0.0155302 | 0.0753331 |
| snn_pred_analog_ns20_plain_ft_seed9_e1500 | snn | 9 | 1500 | 1417.05 | 1390.9 | 1377.07 | 0.0103607 | 0.0163183 | 0.0780867 |

## Model-Wise Mean ± Std

| model | metric | mean | std | n |
|---|---|---:|---:|---:|
| rnn | loss_end | 279.924 | 21.2388 | 2 |
| rnn | tf_mse | 0.00869282 | 0.000609541 | 2 |
| rnn | free_run_mse | 0.00985866 | 0.00137556 | 2 |
| rnn | free_run_pixel_mse | 0.0515256 | 0.0062151 | 2 |
| snn | loss_end | 555.703 | 455.713 | 17 |
| snn | tf_mse | 0.0107668 | 0.00112716 | 17 |
| snn | free_run_mse | 0.0180275 | 0.00394023 | 17 |
| snn | free_run_pixel_mse | 0.0863946 | 0.0178943 | 17 |

Artifacts:
- per_run_metrics.csv
- summary_by_model.csv
- compare_barplots.png