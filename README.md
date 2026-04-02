# FLAG-X-validation
This repository contains source code, scripts, and instructions to reproduce the results presented in *Towards automated gating of clinical flow cytometry data* ([doi](https://doi.org/10.64898/2026.01.10.698765)).

For a standalone implementation of FLAG-X, please refer to [https://github.com/bionetslab/FLAG-X](https://github.com/bionetslab/FLAG-X).

## Version
FLAG-X version 0.2.0 was used to generate the results presented here.

## Data
The immune status (Imstat) and lymphoma (Lt1, Lt2) datasets are not publicly available due to data protection regulations and ethical restrictions.

However, the Flowcyt dataset is available for download [here](https://cuicloud.unige.ch/index.php/s/55PHBLEynrp5pN8). Instructions on how to process the raw data files are provided in the [FlowCyt-Classification-Benchmark repository](https://github.com/VIPER-GENEVA/FlowCyt-Classification-Benchmark).

Running the function `main_data_processing()` from [`main.py`](main.py) produces and saves preprocessed training and test data as *.npy* files in [`data/np_files`](data/np_files).

## Scripts
- [`main.py`](main.py) contains varius scripts to reproduce the experimental results. Each script includes a docstring explaining its purpose.
- [`main_plotting.py`](main_plotting.py) contains varius scripts for generating the figures.
- [`run_experiments.py`](run_experiments.py) offers a command-line interface for running selected scripts from [`main.py`](main.py). It was used to launch jobs on our HPC system through the SLURM scripts in [`slurm_scripts`](slurm_scripts).

## Results
All results from [`main.py`](main.py) and [`main_plotting.py`](main_plotting.py) are saved to [`results`](results). Figures and selected raw outputs are provided for reference.

## Validation

The [`validation`](validation) package contains tree modules that provide functionality for validation:

- [`gating`](validation/gating) with the `GateMeClassClassifier` and `DgcytofClassifier` classes that wrap the GateMeClass [[Caligola *et al.*, 2024](https://doi.org/10.1093/bioinformatics/btae322)] and DGCyTOF [[Cheng *et al.*, 2022](https://doi.org/10.1371/journal.pcbi.1008885)] methods and expose a Scikit-learn style API.
- [`plt`](validation/plt) with helper functions for plotting.
- [`utils`](validation/utils) with utility functions for validation and model evaluation (e.g., performance score computation).

## Environments

- [`environment.yml`](environment.yml) defines the environment used to run the experiments.
- [`environment_cpu_only.yml`](environment_cpu_only.yml) defines a version of the environment where PyTorch is installed in CPU only mode.
- [`gmc_install.txt`](gmc_install.txt) provides install instructions for GateMeClass
- [`environment_windows.yml`](environment_windows.yml) defines an environment for running FLAG-X on a Windows. However, we strongly advise against using this setup due to stability concerns. If Windows is unavoidable we recommend resorting to [WSL](https://learn.microsoft.com/en-us/windows/wsl/install). 




