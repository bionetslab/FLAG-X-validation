# GateMeClass bug report

**Affected runs:**

- our_data, log10_100, w(out) unknowns (gmc0, 1)
- our_data, log10_cw, w(out) unkn (gmc2, 3)

## Error

```
rpy2.rinterface_lib.embedded.RRuntimeError: Error in predict.Mclust(cl, X[-index_sel]) : object not of class 'Mclust'
```

## Call chain

1. `GateMeClass_annotate()` (annotates cells) calls
2. `set_marker_expression()` (sets the marker signature of each cell), which calls
3. `set_marker_expression_GMM(X, ...)` (applies GMM to classify marker expressions, `X` is expression vector of marker)

## Analysis

- Now: if RSS is active, if data negatively skewed then `test` is dominated by low values.
- This call fails:
  ```r
  cl <- Mclust(test, G = 2, verbose = F, modelNames = GMM_parameterization)
  ```
- Probably this fails because data is negatively skewed due to cutoff and all values are the same ...
- Note that if parameterization is `'E'` instead of `'V'` then `test = X`, this probably resolves the error.

## Extended analysis

`GMM_parameterization` sets the variance model of the Gaussian mixture (`Mclust`) that GateMeClass fits per marker to call cells `+`/`-`: `'E'` (Equal) forces both Gaussian components to share one variance, `'V'` (Variable) lets each component have its own. GateMeClass's own default is `'E'`; our wrapper (`validation/gating/gatemeclass.py`) defaults to `gmc_gmm_parameterization = "V"`.

1. **The parameterization also switches ranked set sampling (RSS).** Both `GateMeClass_train()` and `GateMeClass_annotate()` pass
   ```r
   RSS = ifelse(GMM_parameterization == "E", F, T)
   ```
   so `'V'` turns RSS on and `'E'` turns it off.

2. **What the GMM is fitted to (`set_marker_expression_GMM()`).**
   - RSS off (`'E'`): `test <- X`, i.e. all cells' values of that marker.
   - RSS on (`'V'`): random pairs of cells are drawn and one value per pair is kept: the **minimum** if the marker's distribution is negatively skewed (`moments::skewness(X) < 0`), otherwise the maximum.

3. **Why it crashes with cutoff preprocessing (`log10_100`, `log10_cw`).** The cutoff clips low values, so many cells sit at exactly the cutoff value. Taking the minimum of each pair leaves an almost constant `test` vector. `Mclust(test, G = 2, modelNames = "V")` cannot fit two components to that and returns `NULL` instead of a model, so the next line, `predict.Mclust(cl, X[-index_sel])`, raises `object not of class 'Mclust'`.

4. **Why `'E'` probably resolves it.** With RSS off the GMM is fitted to all values (`test = X`), which still vary above the cutoff, and the `predict.Mclust` call (inside the RSS branch) is never reached. Note that this changes the model (equal instead of variable component variances), which should be stated if the affected runs are reported with `'E'`.

*Source: `GateMeClass` R package as installed in the `gmc` conda env (functions `set_marker_expression_GMM`, `set_marker_expression`, `GateMeClass_train`, `GateMeClass_annotate`).*

## Environment notes

pytometry 0.1.6 is latest, mamba only has 0.1.4.

```bash
mamba create -n flowy scanpy python-igraph leidenalg somoclu numba umap-learn scipy scikit-learn seaborn matplotlib=3.6
pip install flowio pytometry torch torchvision torchaudio
pip uninstall matplotlib
mamba uninstall matplotlib
mamba install matplotlib=3.6
```
