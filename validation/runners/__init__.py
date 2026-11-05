"""
Per-method runners (PLAN.md step 4): the method-specific glue between an experiment script and a wrapper / class.

Every runner module has the same small interface and is imported only in its own env
(importlib.import_module('validation.runners.' + method)):
    MODULES: list of module names whose versions are recorded (validation.core.runs.run_metadata)
    input_arm(params) -> 'shared' or 'native': input of a grid point. 'shared' = shared preprocessing (log10 with
        cut-offs), method-internal preprocessing off; 'native' = curated (untransformed) data, method-internal
        preprocessing on (only LDA-Abdelaal and DeepCyTOF, PLAN.md D3)
    build(params, seed, dataset, n_jobs) -> model (unfitted)
    fit(model, X, y, sample_ids) -> fitted model
    predict(model, X, proba=False) -> (y_pred, proba or None); y_pred in the original labels, -1 = abstention
    fit_info(model) -> dict (optional): small JSON-serializable facts about the fitted model, e.g. the number of
        epochs actually trained with early stopping; stored in the experiment outputs
The runners know nothing about splits, downsampling or outputs (experiment scripts) and nothing about other methods.
Keep this package Python 3.5-compatible (deepcytof runs in fx_deepcytof).
"""
