import burin
import numpy as np
import pytest
import shapely
import xarray as xr
import xdggs

import xdggs_burin  # noqa: F401  registers the grid


@pytest.fixture
def ds():
    lon, lat = np.linspace(-0.3, 0.2, 60), np.linspace(51.3, 51.7, 60)
    cells = np.unique(burin.cells_from_lonlat(lon, lat, 8))
    return xr.Dataset(
        {"value": ("cells", np.arange(cells.size, dtype=float))},
        coords={"cell_ids": ("cells", cells, {"grid_name": "rhealpix.burin", "level": 8})},
    )


def test_decode_and_grid_info(ds):
    decoded = xdggs.decode(ds)
    assert isinstance(decoded.dggs.index, xdggs_burin.RHEALPixIndex)
    assert decoded.dggs.grid_info == xdggs_burin.RHEALPixInfo(level=8)


def test_centres_and_boundaries(ds):
    decoded = xdggs.decode(ds)
    centres = decoded.dggs.cell_centers()
    assert set(centres.coords) >= {"longitude", "latitude"}
    boundaries = decoded.dggs.cell_boundaries()
    assert boundaries.sizes["cells"] == ds.sizes["cells"]
    assert shapely.contains_xy(boundaries.values, centres["longitude"].values, centres["latitude"].values).all()


def test_sel_latlon(ds):
    decoded = xdggs.decode(ds)
    lon, lat = decoded.dggs.cell_centers()["longitude"].values[:4], decoded.dggs.cell_centers()["latitude"].values[:4]
    selected = decoded.dggs.sel_latlon(latitude=lat, longitude=lon)
    np.testing.assert_array_equal(selected["cell_ids"].values, ds["cell_ids"].values[:4])


def test_zoom_to(ds):
    decoded = xdggs.decode(ds)
    parents = decoded.dggs.zoom_to(6)
    assert parents.dims == ("cells",)
    children = decoded.dggs.zoom_to(9)
    assert children.dims == ("cells", "children") and children.sizes["children"] == 9


def test_fingerprint_through_the_accessor(ds):
    decoded = xdggs.decode(ds)
    expected = burin.Tree.from_cells(ds["cell_ids"].values, 8).root_hex
    assert decoded.dggs.index.fingerprint() == expected


def test_explore_draws_cells_across_the_antimeridian():
    pytest.importorskip("lonboard")
    cells = np.unique(burin.cells_from_lonlat(np.linspace(170, 190, 50), np.linspace(-5, 5, 50), 5))
    ds = xr.Dataset(
        {"value": ("cells", np.arange(cells.size, dtype=float))},
        coords={"cell_ids": ("cells", cells, {"grid_name": "rhealpix.burin", "level": 5})},
    )
    decoded = xdggs.decode(ds)
    boundaries = decoded.dggs.cell_boundaries().values
    assert ((shapely.bounds(boundaries)[:, 2] - shapely.bounds(boundaries)[:, 0]) < 5).all()
    widget = decoded["value"].dggs.explore()
    assert len(widget.layers) == 1
