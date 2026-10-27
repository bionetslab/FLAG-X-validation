# Additional methods to be added to the benchmark

- flowDensity (https://doi.org/10.1093/bioinformatics/btu677)
- OpenCyto (https://doi.org/10.1371/journal.pcbi.1003806)
- DeepCyTOF (https://doi.org/10.1093/bioinformatics/btx448)
- LDA (https://doi.org/10.1002/cyto.a.23738)
- ACDC (https://doi.org/10.1093/bioinformatics/btx054)
- Bayesian trees (http://proceedings.mlr.press/v85/ji18a/ji18a.pdf)

## Task
Investigate and report all findings in ``additional_methods.md``
- Investigate which of those methods are truly supervised
- Investigate which of those methods have their source code publicly available
- Check if it is possible to wrap this in a sklearn-like API, see ``../validation/gating`` for examples

Implement:
- For those methods for which it is possible 
    - Add file ``method_name.py`` to ``../validation/gating``
    - Implement the wrapper
        - The wrapper should expose all hyperparameters of the underlying method
    - Add file ``method_name.yml`` to ``../envs`` 
            - should contain an env with which the wrapped method can be used
            - numpy should be installed in each env as well

## Rules
- Use one agent per method
- Never use more than 60 CPUs at once, ensure via taskset
- Yo can set up a conda env per method and one additional one if your work requires, document this as well in this md


