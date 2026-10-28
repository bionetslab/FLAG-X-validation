# FLAG-X-validation

This repository contains source code, scripts, and instructions to reproduce the results presented in *FLAG-X: Hybrid machine learning workflows for automated gating of clinical flow cytometry data* ([doi](https://doi.org/10.64898/2026.01.10.698765)).

For a standalone implementation of FLAG-X, please refer to [https://github.com/bionetslab/FLAG-X](https://github.com/bionetslab/FLAG-X).

## Version

FLAG-X version 0.2.1 was used to generate the results presented here.

## Environments
All environments used for the analysis are in [``./envs``](./envs). Details on which environment to use are in the docstring of each script file.

## Data

- The **immune status (Imstat)** and **lymphoma (Lt1, Lt2, Lt1b, Lt2b)** datasets are not publicly available due to data protection regulations and ethical restrictions.

- **Flowcyt:** 
    - Download: [https://cuicloud.unige.ch/index.php/s/55PHBLEynrp5pN8](https://cuicloud.unige.ch/index.php/s/55PHBLEynrp5pN8) 
    - Data processing instructions: [https://github.com/VIPER-GENEVA/FlowCyt-Classification-Benchmark](https://github.com/VIPER-GENEVA/FlowCyt-Classification-Benchmark).
    - Publication: *Bini et al.*, 2024, [https://raw.githubusercontent.com/mlresearch/v248/main/assets/bini24a/bini24a.pdf](https://raw.githubusercontent.com/mlresearch/v248/main/assets/bini24a/bini24a.pdf)

- **BLAST110:**
    - Download: [https://zenodo.org/records/11046402](https://zenodo.org/records/11046402)
    - Publication: Mocking *et al.*, 2024, [https://doi.org/10.1038/s43856-024-00700-x](https://doi.org/10.1038/s43856-024-00700-x) 

## Scripts
\# Todo

## Results
Generated results are saved to [``./results``](./results). Results data necessary to reproduce the figures is provided in this repo, raw outputs such as saved models are not provided due to the memory constraints of GitHub.

## Figures
Generated figures are saved to [``./figures``](./figures).

## Validation

The [``./validation``](./validation) package contains all modules that provide functionality for validation:

- [``gating``](validation/gating) provides Scikit-learn-compatible wrappers for established flow cytometry classification methods:
    - [``gatemeclass.py``](./validation/gating/gatemeclass.py) implements ``GateMeClassClassifier``, which wraps the GateMeClass method (Caligola *et al.*, 2024, [https://doi.org/10.1093/bioinformatics/btae322](https://doi.org/10.1093/bioinformatics/btae322)).
    -  [``dgcytof.py``](./validation/gating/dgcytof.py) implements `DGCyTOFClassifier`, which wrap the DGCyTOF method (Cheng *et al.*, 2022, [https://doi.org/10.1371/journal.pcbi.1008885](https://doi.org/10.1371/journal.pcbi.1008885).





