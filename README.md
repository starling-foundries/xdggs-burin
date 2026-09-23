# xdggs-burin

The rHEALPix discrete global grid for [xdggs](https://github.com/xarray-contrib/xdggs), computed
by [burin-core](https://github.com/starling-foundries/burin-core): cell centres, point lookup,
cell boundaries, zooming between levels, and a 32-byte **fingerprint** of any set of cells.

The grid is the OGC-registered rHEALPix DGGRS (`+proj=rhealpix +lon_0=50 +ellps=WGS84`): six
base cells, each split into nine equal-area children at every level. Geometry agrees with the
reference implementation (rhealpixdggs-py) and the sub-zone order and neighbours with DGGAL, the
OGC API - DGGS reference library.

## Install

```bash
pip install xdggs-burin
```

## Use

```python
import numpy as np
import xarray as xr
import xdggs
import xdggs_burin  # registers the "rhealpix.burin" grid with xdggs

lon = np.linspace(-0.3, 0.2, 100)
lat = np.linspace(51.3, 51.7, 100)
cells = xdggs_burin.RHEALPixInfo(level=8).geographic2cell_ids(lon, lat)

ds = xr.Dataset(
    {"value": ("cells", np.arange(cells.size, dtype=float))},
    coords={"cell_ids": ("cells", cells, {"grid_name": "rhealpix.burin", "level": 8})},
)
ds = xdggs.decode(ds)

ds.dggs.cell_centers()          # longitude and latitude of each cell's nucleus
ds.dggs.cell_boundaries()       # shapely polygons
ds.dggs.sel_latlon(lat[:3], lon[:3])
ds.dggs.zoom_to(6)              # parents; a finer level gives the children
ds.dggs.index.fingerprint()     # 64 hex characters, the same for any dataset over the same cells
```

The fingerprint does not depend on the order of the cells, on duplicates, or on how they were
produced: nine cells that make up a parent hash as the parent. `index.to_dggs_json()` writes the
cells as OGC API - DGGS zone data (DGGS-JSON), and `xdggs_burin.from_dggs_json(doc)` reads such a
document back into a dataset.

## The grid

| attribute | value |
|---|---|
| `grid_name` | `rhealpix.burin` |
| `level` | 0 to 15 (aliases: `resolution`, `refinement_level`); `6 * 9**level` cells |
| `indexing_scheme` | `cid` (the default and only value) |
| `lon_0`, `north_square`, `south_square`, `ellipsoid` | optional; must be `50`, `0`, `0`, `WGS84` |

Cell ids are burin cell ids (`cid`): at level `r` they are the integers from `9**(r + 1)` up to
`15 * 9**r`, nested, so a cell's ancestors and descendants are integer arithmetic and the whole
grid at a level is one contiguous range. They are a different encoding from the zone ids of
other rHEALPix plugins such as `rhealpix.dggal`; the text form (`burin.cid_to_suid`, e.g.
`"Q453"`) is the OGC zone identifier.

Cell boundaries are counterclockwise rings in longitude and latitude. Equatorial cells are their
four corners, since their edges are meridians and parallels; polar cells are densified. A cell
that crosses the antimeridian has longitudes past 180, and each polar cap is closed through its
pole.

## Development

```bash
uv sync --extra test   # uses a burin-core checkout at ../burin-core
uv run pytest
```

## License

Apache-2.0.
