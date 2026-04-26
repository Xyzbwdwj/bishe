Scripts used to reproduce figures in:
https://www.biorxiv.org/content/10.1101/2022.05.19.492731v2

## Repository Layout

- `train_predict/`: training, prediction, and evaluation entry points.
- `core/`: model and utility functions (`RNN_Class.py`, `helper.py`).
- `figures/`: non-MNIST figure-specific plotting / experiment scripts (`Figure*.py`, `Figure3.m`, `Figure4_run.sh`, `IO_plot.py`).
- `mnist/`: MNIST data preparation scripts, public comparison scripts, processed inputs, and prediction outputs.
- `data/`: compact project inputs tracked in Git (custom `.pth.tar` and PCA file).
- `Localization/`: trajectory generation and localization preprocessing.
- `NeuralEvidence/`: Allen data access helpers and MATLAB analysis scripts for Fig.2.
- `mnist/compare/public/`: MNIST public comparison outputs and report tables.
- `docs/PROJECT_STRUCTURE.md`: concise directory map and version-control policy.

## GitHub Submission Notes

- Training outputs and checkpoints are intentionally excluded from version control (`Elman_SGD/`, `official_snn/`, `_bench_cpu_*`, `_smoke*`, `logs/`).
- Downloaded and processed MNIST files (`mnist/data/mnist_torchvision/`, `mnist/data/processed/`) are excluded; rerun `python mnist/scripts/Figure6_InputPrep.py` to regenerate.
- `mnist/compare` generated run artifacts (`*.pth.tar`, run `*.txt`, run `*.png`) and `mnist/mnist_next_frame*` prediction images are excluded; keep scripts and summary tables for reproducibility.

# NeuralEvidence (Fig.2)
* DataAccess.py: specs to download data from Allen Brain Observatory; a full guide is available at https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
* Main_clean.m: plot Fig.2BCD.

# Network training and analysis

* Build a python environment:  
`conda env create -f environment.yml`

## Simulation of predictive network (Fig.3)

1. Train the models:  
* Non-predictive model  
`python train_predict/Main_clean.py --input data/SeqN1T100.pth.tar --ae 1 -n 1 --fixo 3 --fixi 3 --rnn_act relu --ac_output sigmoid --savename Elman_SGD/Sigmoid/SeqN1T100_relu_fixio3 `  
* Predictive model  
`python train_predict/Main_clean.py --input data/SeqN1T100.pth.tar --ae 1 -n 1 --fixi 3 --fixo 3 --pred 1 --rnn_act relu --ac_output sigmoid --savename Elman_SGD/Sigmoid/SeqN1T100_pred_relu_fixio3`  

2. Convert python output to *.mat  
`python py2mat_exe.py`

3. Plot the figure:  
`MATLAB figures/Figure3.m`

## Simulation of CA1 and CA3 place cells (Fig.4)

1. Train the models: 
`bash figures/Figure4_run.sh`

2. Plot the figures:  
`python figures/Figure4.py`

## OOM-safe SNN long training (50k)

Run the preconfigured anti-OOM batch script:
`bash train_predict/train_all_snn.sh`


## Localization (Fig.5)
1. Traj_Generate.m: simulation code for straight line exploration (Main reference: https://www.pnas.org/doi/10.1073/pnas.2018422118)  

2. Localization_clean.py: generate model input from simulated trajectories.  

3. Train the model:  
* Non-predictive model  
`for i in {1..10}  
do  
python train_predict/Main_s4.py --epochs 30000 --batch-size 5 --hidden-n 500 --net ElmanRNN --act sigmoid --rnn_act relu --gpu 1 --Hregularized 5.0 --clip 10 --ae 1 --input data/InputNs50_SeqN100_StraightTraj_Marcus_v2.pth.tar --savename Elman_SGD/GridInput/BatchTraining/PhysicalInput_v2/repeats/InputNs50_SeqN100_Marcus_HN500_H5.0_rep$i  
done `

* Predictive model  
`for i in {1..10}  
do  
python train_predict/Main_s4.py --epochs 30000 --batch-size 5 --hidden-n 500 --net ElmanRNN_tp1 --pred 1 --act sigmoid --rnn_act relu --gpu 4 --Hregularized 5.0 --clip 10 --ae 1 --input data/InputNs50_SeqN100_StraightTraj_Marcus_v2.pth.tar --savename Elman_SGD/GridInput/BatchTraining/PhysicalInput_v2/repeats/InputNs50_SeqN100_Marcus_HN500_H5.0_tp1_rep$i
done`

4. Plot the figure:
`python figures/Figure5.py`

## MNIST sequence learning (Fig.6)

1. Generate MNIST inputs:  
`python mnist/scripts/Figure6_InputPrep.py`

2. Train models:  
`python train_predict/Main.py  -n 68 --input mnist/data/processed/MNIST_68PC_SeqN100_Ns5.pth.tar --lr 0.0002 --pred 1 --partial 0.2 --ac_output tanh --Hregularized 1 --epochs 10000 -p 100 --savename mnist/compare/public/MNIST_68PC_SeqN100_Ns5_partial`

3. Plot the figure:  
`python mnist/scripts/Figure6.py`

## Local RNN training (Fig.7)

1. Train models:  
`python train_predict/Main_local.py --n 68 --hidden_n 100 --inputfile 'data/data_pca.pkl' --savename 'mnist_local' --learning_alg 'local' --epochs 500001 --lr 0.001 --t 10`

2. Plot the figure:  
`python mnist/scripts/Figure7.py`
