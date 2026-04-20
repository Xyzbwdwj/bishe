# Project Structure

## Source Code
- `Main.py`, `Main_clean.py`, `Main_s4.py`, `Main_local.py`: training entry points.
- `RNN_Class.py`, `helper.py`: model and utilities.
- `Figure3.py`, `Figure4.py`, `Figure5.py`, `Figure6.py`, `Figure7.py`, `Figure6_InputPrep.py`, `IO_plot.py`: figure scripts.
- `Figure3.m`, `Localization/Traj_Generate.m`, `NeuralEvidence/*.m`: MATLAB analysis scripts.

## Data
- `data/*.pth.tar`, `data/data_pca.pkl`: compact inputs required by scripts.
- `data/mnist_torchvision/`: downloaded raw MNIST cache (ignored, regenerate locally).

## Task-Specific Modules
- `Localization/`: localization trajectory generation and preprocessing.
- `NeuralEvidence/`: Allen data access and neural evidence analysis.
- `mnist_compare/public/`: public MNIST comparison scripts and summary reports.

## Generated Outputs (Not Versioned)
- `Elman_SGD/`, `official_snn/`, `_bench_cpu_*/`, `_smoke*`, `logs/`.
- `mnist_compare` run artifacts (`*.pth.tar`, run `*.txt`, run `*.png`).

## Environment
- `environment.yml`: reproducible conda environment.
- `.venv/`: local virtual environment (ignored).
