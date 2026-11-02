"""Training-free decomposition-recomposition calibration on cached CLIP features.

The package is deliberately split in two:

* ``fdrfreq.extract`` needs torch, the project's CLIP copy and the datasets.
  It runs once per dataset/backbone and writes a feature bank directory.
* Everything else is pure numpy. Methods, base-only hyper-parameter selection,
  benchmark evaluation, diagnostics and paper tables operate on the bank, so a
  full ablation matrix runs on a CPU in minutes and is bit-for-bit repeatable.

Protocol rule enforced by the CLI: hyper-parameters are chosen only on
pseudo-incremental episodes built from base-session training images
(``fdrfreq.run select``). Benchmark test images are scored only by
``fdrfreq.run evaluate`` with a locked configuration file.
"""

__all__ = ['bank', 'protocol', 'pipeline', 'metrics', 'select']
__version__ = '0.1.0'
