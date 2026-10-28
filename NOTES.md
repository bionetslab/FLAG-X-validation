# General notes towards a complete plan

- Save raw, compensated data and labels as ``.npy`` files? (-> no need for fcs/csv loading, slimmer deps in method specific envs) or save as ``csv``?, if ``csv`` add label col and harmonize channel names, can choose channels later on
- Isolated envs: one for each method/method-group, analysis step (e.g. data processing, running bechmarks, aggregating results, plotting, ...)
- One script-file per step/task -> define parameters at the top as constants, e.g. channels to be saved
- k-fold CV (nested for tuning) on sample level, generate the splits beforehand, save them somewhere, central source of truth
- Imstat is used for tuning -> save nested splits (train-val) for each outer split (train-test)
- Downsample on the fly with fixed seeds
- Need general benchmark framework with few dependencies (preferably only numpy for data loading, splits are already generated; and of course the classifier deps)
- extensibility on two axis:
    - new methods
    - new datasets
- need to think about what can be varied and how to ensure extensibility without overcomplicating
    - e.g. leave out a specific channel, different preprocessing, ...
- Same splits as used for benchmarking (and tuning) will be reused everwhere
    - Num samples experiments
- Set up all experiments such that they can be run sequentially or via an array job on HPC
- Create a validation package with scikit-learn-like API for each tested method
- Small steps, save intermediate results
    - Save data as ``.npy``/or ``csv``?, if ``csv`` add label col and harmonize channel names.
    - Generate and save splits -> train test for all datasets, additionally train-val for each train-test for for Imstat.
    - Tune models for each trsin-test split of imstat -> respective best param combination; Should one compute a concensus param combinatons, i.e. also average scores across train-test splits?
    - Load data, select channels, preprocess (transform + scaling), fit model -> save trained model, save preprocessor
    - Predict -> save predicted probabilities
    - Score computation/analysis -> save score tables/analysis results
    - Plot -> save Figures













