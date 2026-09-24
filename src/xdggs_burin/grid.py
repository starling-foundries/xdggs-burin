"""The rHEALPix grid for xdggs, computed by burin-core.

The grid is the OGC-registered rHEALPix DGGRS (``+proj=rhealpix +lon_0=50 +ellps=WGS84``, both
polar squares in column 0). Cell ids are burin cids at one level: at level ``r`` they fill
``[9**(r + 1), 15 * 9**r)``, nested, so ancestors and descendants are integer arithmetic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import burin
import numpy as np
from xdggs.grid import DGGSInfo, translate_parameters

GRID_NAME = "rhealpix.burin"
INDEXING_SCHEME = "cid"

#: The grid parameters of the OGC-registered DGGRS; attributes may state them, never change them.
OGC_PARAMETERS = {"lon_0": 50.0, "north_square": 0, "south_square": 0, "ellipsoid": "WGS84"}

#: Points per edge on polar cells; equatorial cells are their four corners.
POLAR_EDGE_POINTS = 5


def _level(value) -> int:
    """A refinement level from an attribute: an int, or a float or string that is one exactly."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"level must be an integer, got {value!r}")
    try:
        number = value if isinstance(value, (int, np.integer)) else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"level must be an integer, got {value!r}") from None
    if number != int(number):
        raise ValueError(f"level must be an integer, got {value!r}")
    return int(number)


def _ellipsoid_name(value) -> str:
    if isinstance(value, str):
        return value.upper()
    if isinstance(value, dict) and "name" in value:
        return str(value["name"]).upper()
    return repr(value)


def polygons_shapely(coords: np.ndarray, offsets: np.ndarray):
    import shapely

    return shapely.from_ragged_array(shapely.GeometryType.POLYGON, coords, (offsets, np.arange(len(offsets))))


def polygons_geoarrow(coords: np.ndarray, offsets: np.ndarray):
    import pyproj
    from arro3.core import list_array

    ring_offsets = offsets.astype("int32")
    geom_offsets = np.arange(len(offsets), dtype="int32")
    polygons = list_array(geom_offsets, list_array(ring_offsets, coords))
    crs = pyproj.CRS.from_epsg(4326)
    return polygons.cast(
        polygons.field.with_metadata(
            {
                "ARROW:extension:name": "geoarrow.polygon",
                "ARROW:extension:metadata": json.dumps({"crs": crs.to_json_dict()}),
            }
        )
    )


@dataclass(frozen=True)
class RHEALPixInfo(DGGSInfo):
    """Grid information for the rHEALPix DGGS computed by burin-core.

    Parameters
    ----------
    level : int
        Refinement level, 0 to 15. There are ``6 * 9**level`` cells, all of equal area.
    ellipsoid : str, default "WGS84"
        Always WGS84: the grid is the OGC-registered rHEALPix DGGRS.
    """

    level: int
    """int : The refinement level of the grid"""

    ellipsoid: str = "WGS84"

    valid_parameters: ClassVar[dict[str, Any]] = {"level": range(burin.MAX_LEVEL + 1)}

    def __post_init__(self):
        if self.level not in self.valid_parameters["level"]:
            raise ValueError(f"level must be an integer between 0 and {burin.MAX_LEVEL}")
        if _ellipsoid_name(self.ellipsoid) != "WGS84":
            raise ValueError(f"the {GRID_NAME} grid is on the WGS84 ellipsoid, not {self.ellipsoid!r}")

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        """Construct a `RHEALPixInfo` from a mapping of attributes.

        ``level`` may also be given as ``resolution`` or ``refinement_level``. The grid parameters
        ``lon_0``, ``north_square``, ``south_square`` and ``ellipsoid`` are accepted only at the
        values of the OGC-registered grid; any other value raises ``ValueError``.
        """
        params = dict(mapping)
        scheme = params.pop("indexing_scheme", INDEXING_SCHEME)
        if scheme != INDEXING_SCHEME:
            raise ValueError(f"unknown indexing scheme {scheme!r}; {GRID_NAME} ids are {INDEXING_SCHEME!r}")
        for key, expected in OGC_PARAMETERS.items():
            if key not in params:
                continue
            value = params.pop(key)
            same = _ellipsoid_name(value) == expected if key == "ellipsoid" else value == expected
            if not same:
                raise ValueError(f"{GRID_NAME} is the OGC rHEALPix grid: {key} must be {expected!r}, got {value!r}")
        translations = {
            "resolution": ("level", _level),
            "refinement_level": ("level", _level),
            "level": ("level", _level),
        }
        return cls(**translate_parameters(params, translations))

    def to_dict(self: Self) -> dict[str, Any]:
        """The normalized grid parameters."""
        return {"grid_name": GRID_NAME, "level": self.level, "indexing_scheme": INDEXING_SCHEME}

    def cell_ids2geographic(self, cell_ids) -> tuple[np.ndarray, np.ndarray]:
        """The nucleus of each cell as ``(lon, lat)`` in degrees."""
        return burin.cells_to_lonlat(burin.as_cell_ids(cell_ids, self.level))

    def geographic2cell_ids(self, lon, lat) -> np.ndarray:
        """The cell containing each point (degrees), as uint64 cell ids."""
        return burin.cells_from_lonlat(lon, lat, self.level)

    def cell_boundaries(self, cell_ids, backend="shapely"):
        """Cell boundary polygons.

        Rings are counterclockwise in ``(lon, lat)`` degrees. Equatorial cells are their four
        corners, whose edges are meridians and parallels; polar cells are densified. A cell that
        crosses the antimeridian has longitudes past 180, and each polar cap is closed through its
        pole.

        Parameters
        ----------
        cell_ids : array-like
            The cell ids.
        backend : {"shapely", "geoarrow"}, default: "shapely"
            The backend to convert to.
        """
        backends = {"shapely": polygons_shapely, "geoarrow": polygons_geoarrow}
        backend_func = backends.get(backend)
        if backend_func is None:
            raise ValueError(f"invalid backend: {backend!r}")
        ids = burin.as_cell_ids(cell_ids, self.level).ravel()
        coords, offsets = burin.cell_boundaries(ids, n=POLAR_EDGE_POINTS)
        return backend_func(coords, offsets)

    def zoom_to(self, cell_ids, level: int) -> np.ndarray:
        """Ancestors at a coarser ``level``, or descendants (an extra trailing axis of
        ``9**(level - self.level)``, in nested order) at a finer one."""
        return burin.zoom_to(cell_ids, self.level, level)
