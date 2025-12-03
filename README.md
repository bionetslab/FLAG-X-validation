# FLAG-X-validation
This repository contains source code, scripts, and instructions to reproduce the results presented in *Towards automated gating of clinical flow cytometry data* ([doi](add doi)).

For a standalone implementation of FLAG-X, please refer to [https://github.com/bionetslab/FLAG-X](https://github.com/bionetslab/FLAG-X).

## Data
The immune status (Imstat) and lymphoma (Lt1, Lt2) datasets are not publicly available due to data protection regulations and ethical restrictions.

However, the Flowcyt dataset is available for download [here](https://cuicloud.unige.ch/index.php/s/55PHBLEynrp5pN8). Instructions on how to process the raw data files are provided in the [FlowCyt-Classification-Benchmark repository](https://github.com/VIPER-GENEVA/FlowCyt-Classification-Benchmark).

Running the function `main_data_processing()` from [`main.py`](main.py) produces and saves preprocessed training and test data as *.npy* files in [data/np_files](data/np_files).

## Scripts
- [`main.py`](main.py) contains varius scripts to reproduce the experimental results. Each script includes a docstring explaining its purpose.
- [`main_plotting.py`](main_plotting.py) contains varius scripts for generating the figures used in the analysis.
- [`run_experiments.py`](run_experiments.py) offers a simple command-line interface for running selected scripts from [`main.py`](main.py). It was used to launch jobs on our HPC system through the SLURM scripts in [slurm_scripts](slurm_scripts).




