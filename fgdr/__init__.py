"""FGDR: training-free frequency-grounded decomposition-recomposition for CLIP FSCIL.

The package works on cached frozen-CLIP features, so every calibration rule is a
closed-form NumPy computation that can be swept on base-only pseudo-incremental
episodes before a single locked evaluation on the benchmark test split.

Modules
-------
cache      benchmark feature cache (train/test kept in separate files)
protocol   real benchmark episodes and base-only pseudo-incremental episodes
bimc_ref   exact re-implementation of BiMC / BiMC-dagger scoring on cached tensors
recompose  TEEN/BiMC holistic calibration, text-weighted donors, segment recomposition
grounding  frequency-grounded channel masks, keyword library, mask diagnostics
anchor     shot-aware pooled within-class discriminant and probability mixing
keywords   lexical keyword extraction for the FDR-style keyword library
metrics    session metrics, error decomposition, paired statistics
pipeline   method specification, episode runner, grid expansion and selection
"""

__version__ = '0.1.0'
