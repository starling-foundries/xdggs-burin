"""The rHEALPix DGGS for xdggs, computed by burin-core.

Importing this package registers the grid ``rhealpix.burin`` with xdggs.
"""
from importlib.metadata import version

from xdggs_burin.grid import GRID_NAME, RHEALPixInfo
from xdggs_burin.index import RHEALPixIndex, from_dggs_json

__version__ = version("xdggs-burin")

__all__ = ["GRID_NAME", "RHEALPixInfo", "RHEALPixIndex", "from_dggs_json", "__version__"]
