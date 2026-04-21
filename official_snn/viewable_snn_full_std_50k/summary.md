# SNN Validity Summary

- Checkpoint: `official_snn/snn_full_std_50k.pth.tar`
- Model: `ElmanSNN`
- Epochs: `50000`
- Final loss: `0.049546`
- Min loss: `0.024582`
- Eval MSE: `0.000001`
- Eval RMSE: `0.000963`
- Eval R2: `0.970215`
- Global spike rate: `0.098350`
- Active neuron fraction (rate>0.01): `0.4300`

## Generated files
- `checkpoint_summary.json`
- `training_loss.csv`
- `training_loss_raw_vs_smooth.png`
- `spike_raster_sample0.png`
- `spike_rate_hist.png`
- `robustness_noise_sweep.csv`
- `robustness_noise_sweep.png`
- `state_dict_keys.txt`
- `state_dict_shapes.csv`
- `prediction_last_step_sample.csv`
- `prediction_last_step_scatter.png`
- `prediction_last_step_residual_hist.png`
- `prediction_time_mean_sample.csv`
- `prediction_time_mean_scatter.png`
- `prediction_time_mean_residual_hist.png`
- `prediction_metrics.txt`

## Additional prediction metrics
- Last-step readout: `R2=0.961486`, `RMSE=0.001096`, `MAE=0.000656`
- Time-mean readout: `R2=0.981299`, `RMSE=0.000763`, `MAE=0.000513`
