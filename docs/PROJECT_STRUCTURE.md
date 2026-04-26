# Project Structure

## Source Code
- `train_predict/`: general training entry points (`Main.py`, `Main_clean.py`, `Main_s4.py`, `Main_local.py`, `train_all_snn.sh`).
- `core/`: model and utility functions (`RNN_Class.py`, `helper.py`).
- `figures/`: non-MNIST figure scripts (`Figure3.py`, `Figure4.py`, `Figure5.py`, `IO_plot.py`, `Figure4_run.sh`, `Figure3.m`).
- `mnist/`: MNIST data preparation, figure scripts, public comparison scripts, and prediction outputs.
- `Localization/Traj_Generate.m`, `NeuralEvidence/*.m`: MATLAB analysis scripts.

## Data
- `data/*.pth.tar`, `data/data_pca.pkl`: compact inputs required by scripts.
- `mnist/data/`: MNIST raw cache and processed PCA sequence inputs.

## Task-Specific Modules
- `Localization/`: localization trajectory generation and preprocessing.
- `NeuralEvidence/`: Allen data access and neural evidence analysis.
- `mnist/compare/public/`: public MNIST comparison outputs and summary reports.

## Generated Outputs (Not Versioned)
- `Elman_SGD/`, `official_snn/`, `_bench_cpu_*/`, `_smoke*`, `logs/`.
- `mnist/data/processed/`: generated MNIST PCA inputs and raw arrays.
- `mnist/compare` run artifacts and `mnist/mnist_next_frame*` prediction images.

## Environment
- `environment.yml`: reproducible conda environment.
- `.venv/`: local virtual environment (ignored).
