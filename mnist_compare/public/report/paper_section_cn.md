# 4.X 公开数据集实验与对比分析（MNIST 序列预测）

## 4.X.1 实验目的
为验证所设计脉冲神经网络在公开数据集上的序列预测能力与稳定性，本节在 MNIST 数据集上构建序列预测任务，并与非脉冲基线（RNN-Pred）进行对比。

## 4.X.2 实验设置
- 数据集：MNIST（训练集 60,000 张，公开数据集）。
- 预处理：将 28×28 图像展平后进行 PCA 降维（68 维），构造长度为 100 的序列，每批次 5 条序列。
- 任务形式：一步预测（`--pred 1`）。
- 对比模型：
  - `RNN-Pred`：基于 `ElmanRNN_pred`；
  - `SNN-Pred`：基于 `ElmanSNN_pred`（LIF 膜电位更新 + surrogate gradient）。
- 训练配置（两模型保持一致）：`epochs=3000`，`Adam`，`lr=1e-3`，`ac_output=tanh`，CPU 训练。
- 随机重复：3 个种子（0/1/2），报告均值±标准差。
- 评估指标：
  - `tf_mse`：teacher-forcing 条件下 MSE；
  - `free_run_mse`：自由滚动预测阶段（stop_t=17 之后）MSE；
  - `free_run_pixel_mse`：反 PCA 回像素空间后的自由滚动 MSE。

## 4.X.3 对比结果

| 模型 | loss_end (mean±std) | tf_mse (mean±std) | free_run_mse (mean±std) | free_run_pixel_mse (mean±std) |
|---|---:|---:|---:|---:|
| RNN-Pred | 135.614 ± 138.319 | 0.004457 ± 0.004049 | 0.009062 ± 0.001533 | 0.044682 ± 0.009736 |
| SNN-Pred | 98.619 ± 8.983 | 0.003094 ± 0.000296 | 0.003650 ± 0.000405 | 0.018911 ± 0.001852 |

相对 RNN-Pred，SNN-Pred 的指标改善幅度为：
- `tf_mse` 降低约 **30.60%**；
- `free_run_mse` 降低约 **59.72%**；
- `free_run_pixel_mse` 降低约 **57.68%**；
- `loss_end` 降低约 **27.28%**。

## 4.X.4 结果分析
1. 在自由滚动预测指标（`free_run_mse` 与 `free_run_pixel_mse`）上，SNN-Pred 明显优于 RNN-Pred，说明引入脉冲动力学后，对长时间序列的自回归预测更稳定。  
2. SNN-Pred 在 3 个随机种子上的方差更小（尤其 `loss_end` 与 `free_run_pixel_mse`），表明其训练结果一致性更好。  
3. 本实验支持“具备生物启发记忆机制的脉冲网络可以提升序列预测性能”的结论。

## 4.X.5 复现与产物
- 一键脚本：`mnist_compare/public/run_public_mnist_compare.sh`
- 评估脚本：`mnist_compare/public/evaluate_public_mnist_compare.py`
- 逐次指标：`mnist_compare/public/report/per_run_metrics.csv`
- 汇总指标：`mnist_compare/public/report/summary_by_model.csv`
- 可视化：`mnist_compare/public/report/compare_barplots.png`

