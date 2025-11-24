"""lrs2sky: Sky modeling tools for HET/LRS2.

This package provides modules to:
- Ingest LRS2 FITS files and build an index (ingest)
- Compute metadata labels from FITS headers (labels)
- Extract and normalize sky spectra (spectrum)
- Build a low-dimensional basis and regression models (model)

These are lightweight, initial implementations intended to be extended.
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "__version__",
]

try:
    __version__ = version("lrs2sky")
except PackageNotFoundError:
    __version__ = "0.0.0"
