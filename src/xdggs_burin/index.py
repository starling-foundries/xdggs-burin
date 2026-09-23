"""The xdggs index for ``rhealpix.burin`` cell ids, with a fingerprint of the cells it holds."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import burin
import numpy as np
import xarray as xr
from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs

from xdggs_burin.grid import GRID_NAME, RHEALPixInfo

#: Deepest sub-zone depth written to one DGGS-JSON document (9**9 values).
MAX_ZONE_DEPTH = 9


@register_dggs(GRID_NAME)
class RHEALPixIndex(DGGSIndex):
    _grid: RHEALPixInfo

    def __init__(self, cell_ids: Any | xr.Index, dim: str, name: str, grid_info: RHEALPixInfo):
        super().__init__(cell_ids, dim, grid_info)
        self._name = name
        self._index.index.name = name

    @classmethod
    def from_variables(cls, variables: Mapping[Any, xr.Variable], *, options: Mapping[str, Any]) -> RHEALPixIndex:
        name, var, dim = _extract_cell_id_variable(variables)
        grid_info = RHEALPixInfo.from_dict(var.attrs | dict(options or {}))
        return cls(var.data, dim, name, grid_info)

    @classmethod
    def full_domain(cls, level: int, dim: str = "cells", name: str = "cell_ids", *, options=None) -> RHEALPixIndex:
        """Every cell of the grid at ``level``, sorted."""
        grid_info = RHEALPixInfo.from_dict({"level": level} | dict(options or {}))
        return cls(burin.full_domain(grid_info.level), dim, name, grid_info)

    @property
    def grid_info(self) -> RHEALPixInfo:
        return self._grid

    def _replace(self, new_index: xr.Index):
        return type(self)(new_index, self._dim, self._name, self._grid)

    def _repr_inline_(self, max_width: int):
        return f"RHEALPixIndex(level={self._grid.level})"

    def _cells(self) -> np.ndarray:
        return burin.as_cell_ids(self.values(), self._grid.level)

    def fingerprint(self) -> str:
        """The 32-byte root of the set of cells, as 64 hex characters.

        Order and duplicates do not matter, and nine cells that make up a parent hash as the
        parent, so any two datasets over the same cells at the same level have the same
        fingerprint, however they were produced.
        """
        return burin.Tree.from_cells(self._cells(), self._grid.level).root_hex

    def to_dggs_json(self, zone: int | None = None, field: str = "coverage") -> dict:
        """The cells as OGC API - DGGS zone data (DGGS-JSON) over the sub-zones of ``zone``.

        ``zone`` defaults to the smallest zone containing every cell. Values are 1 at the cells
        held and null elsewhere, in the DGGRS's sub-zone order.
        """
        cells = self._cells()
        level = self._grid.level
        if cells.size == 0:
            raise ValueError("no cells to encode")
        if zone is None:
            zone = _common_ancestor(cells, level)
        zone_level = int(burin.cell_levels([zone])[0])
        if zone_level > level:
            raise ValueError(f"zone is at level {zone_level}, below the cells at level {level}")
        if level - zone_level > MAX_ZONE_DEPTH:
            raise ValueError(f"cells are {level - zone_level} levels below the zone; at most {MAX_ZONE_DEPTH} fit one document")
        if not np.all(burin.zoom_to(cells, level, zone_level) == np.uint64(zone)):
            raise ValueError(f"not every cell lies in zone {burin.cid_to_suid(int(zone))}")
        return burin.Tree.from_cells(cells, level).to_dggs_json(int(zone), field)

    @classmethod
    def from_dggs_json(cls, doc: Mapping, *, field: str | None = None, dim: str = "cells", name: str = "cell_ids") -> RHEALPixIndex:
        """An index over the sub-zones a DGGS-JSON document gives a value for."""
        tree = burin.Tree.from_dggs_json(doc, field=field)
        cells = np.asarray(tree.leaves(), dtype=np.uint64)
        return cls(cells, dim, name, RHEALPixInfo(level=tree.depth))


def _common_ancestor(cells: np.ndarray, level: int) -> int:
    for zone_level in range(level, -1, -1):
        zones = np.unique(burin.zoom_to(cells, level, zone_level))
        if zones.size == 1:
            return int(zones[0])
    raise ValueError("the cells span more than one base cell; pass zone")


def from_dggs_json(doc: Mapping, *, field: str | None = None, dim: str = "cells", name: str = "cell_ids") -> xr.Dataset:
    """A dataset whose ``cell_ids`` are the sub-zones a DGGS-JSON document gives a value for,
    decoded with the ``rhealpix.burin`` index."""
    import xdggs

    index = RHEALPixIndex.from_dggs_json(doc, field=field, dim=dim, name=name)
    ds = xr.Dataset(coords={name: (dim, index.values(), index.grid_info.to_dict())})
    return xdggs.decode(ds, name=name)
