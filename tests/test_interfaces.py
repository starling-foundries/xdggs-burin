"""The paths real data takes into and out of the grid: files, xdggs's own encodings, attributes as
they come off disk, cell id dtypes, subsetting, and the geometry as a whole."""
import numpy as np
import pytest
import shapely
import xarray as xr
import xdggs
from xdggs.utils import GRID_REGISTRY

import burin
from xdggs_burin import GRID_NAME, RHEALPixIndex, RHEALPixInfo


def _cells(level, n=80):
    lon, lat = np.linspace(-0.3, 0.2, n), np.linspace(51.3, 51.7, n)
    return np.unique(burin.cells_from_lonlat(lon, lat, level))


def _ds(cells, level):
    return xr.Dataset(
        {"value": ("cells", np.arange(len(cells), dtype=float))},
        coords={"cell_ids": ("cells", cells, {"grid_name": GRID_NAME, "level": level})},
    )


def test_registration_adds_the_grid_without_replacing_others():
    assert GRID_REGISTRY[GRID_NAME] is RHEALPixIndex
    assert {"h3", "healpix"} <= set(GRID_REGISTRY)


def test_netcdf4_keeps_deep_ids_exactly(tmp_path):
    pytest.importorskip("h5netcdf")
    cells = _cells(12)
    assert int(cells.max()) > 2**31, "ids at level 12 do not fit 32 bits"
    path = tmp_path / "cells.nc"
    _ds(cells, 12).to_netcdf(path, engine="h5netcdf")
    back = xr.open_dataset(path, engine="h5netcdf")
    assert back["cell_ids"].dtype == np.uint64
    np.testing.assert_array_equal(back["cell_ids"].values, cells)
    decoded = xdggs.decode(back)
    assert decoded.dggs.grid_info == RHEALPixInfo(level=12)
    assert decoded.dggs.index.fingerprint() == burin.Tree.from_cells(cells, 12).root_hex


def test_netcdf3_refuses_ids_it_cannot_hold(tmp_path):
    pytest.importorskip("scipy")
    with pytest.raises(ValueError, match="safely cast"):
        _ds(_cells(12), 12).to_netcdf(tmp_path / "deep.nc", engine="scipy")
    shallow = _cells(8)
    _ds(shallow, 8).to_netcdf(tmp_path / "shallow.nc", engine="scipy")
    back = xdggs.decode(xr.open_dataset(tmp_path / "shallow.nc", engine="scipy"))
    assert back["cell_ids"].dtype == np.int32, "netCDF3 stores them as int32"
    assert back.dggs.index.fingerprint() == burin.Tree.from_cells(shallow, 8).root_hex


@pytest.mark.parametrize("convention", ["xdggs", "cf"])
def test_encode_and_decode_round_trip(convention):
    decoded = xdggs.decode(_ds(_cells(10), 10))
    encoded = decoded.dggs.encode(convention=convention)
    again = xdggs.decode(encoded, convention=convention)
    assert again.dggs.grid_info == decoded.dggs.grid_info
    assert again.dggs.index.fingerprint() == decoded.dggs.index.fingerprint()


@pytest.mark.parametrize("level", ["8", np.int32(8), np.int64(8), 8.0, np.float32(8.0)])
def test_levels_as_they_come_off_disk(level):
    assert xdggs.decode(_ds(_cells(8), level)).dggs.grid_info.level == 8


@pytest.mark.parametrize("level", [8.5, "8.5", np.float32(8.5), "eight", True, None])
def test_levels_that_are_not_integers_are_refused(level):
    with pytest.raises(ValueError, match="level"):
        RHEALPixInfo.from_dict({"grid_name": GRID_NAME, "level": level})


@pytest.mark.parametrize("dtype", ["u8", "i8", "i4", ">u8"])
def test_integer_ids_of_any_width_and_byte_order(dtype):
    cells = _cells(8)
    decoded = xdggs.decode(_ds(cells.astype(dtype), 8))
    np.testing.assert_array_equal(decoded.dggs.cell_centers()["latitude"].values, burin.cells_to_lonlat(cells)[1])
    assert decoded.dggs.index.fingerprint() == burin.Tree.from_cells(cells, 8).root_hex


def test_ids_that_are_not_cells_at_the_level_fail_loudly():
    cells = _cells(8)
    with pytest.raises(TypeError, match="integers"):
        xdggs.decode(_ds(cells.astype("f8"), 8)).dggs.cell_centers()
    parent = np.uint64(burin.suid_to_cid("Q4"))
    mixed = np.concatenate([cells, np.array([parent], dtype=np.uint64)])
    decoded = xdggs.decode(_ds(mixed, 8))
    for use in (decoded.dggs.cell_centers, decoded.dggs.cell_boundaries, decoded.dggs.index.fingerprint):
        with pytest.raises(ValueError, match=f"{int(parent)} is not a cell at level 8"):
            use()


def test_subsets_keep_the_index_and_their_own_fingerprint():
    cells = _cells(9)
    decoded = xdggs.decode(_ds(cells, 9))
    part = decoded.isel(cells=slice(3, 20))
    assert isinstance(part.dggs.index, RHEALPixIndex) and part.dggs.grid_info == decoded.dggs.grid_info
    assert part.dggs.index.fingerprint() == burin.Tree.from_cells(cells[3:20], 9).root_hex
    picked = decoded.sel(cell_ids=cells[[0, 5, 7]])
    assert picked.dggs.index.fingerprint() == burin.Tree.from_cells(cells[[0, 5, 7]], 9).root_hex
    with pytest.raises(KeyError):
        decoded.dggs.sel_latlon(latitude=[-40.0], longitude=[100.0])


def test_the_same_cells_from_different_sources_share_a_fingerprint():
    lon, lat = np.linspace(-0.3, 0.2, 500), np.linspace(51.3, 51.7, 500)
    from_points = xdggs.decode(_ds(np.unique(RHEALPixInfo(level=7).geographic2cell_ids(lon, lat)), 7))
    shuffled = from_points.isel(cells=np.random.default_rng(1).permutation(from_points.sizes["cells"]))
    doc = from_points.dggs.index.to_dggs_json()
    from_json = RHEALPixIndex.from_dggs_json(doc)
    prints = {from_points.dggs.index.fingerprint(), shuffled.dggs.index.fingerprint(), from_json.fingerprint()}
    assert len(prints) == 1


def test_zoom_refuses_expansions_it_cannot_hold():
    decoded = xdggs.decode(_ds(burin.full_domain(1), 1))
    with pytest.raises(ValueError, match="MAX_ZOOM_CELLS"):
        decoded.dggs.zoom_to(12)


@pytest.mark.parametrize("backend", ["shapely", "geoarrow"])
def test_the_whole_grid_tiles_the_globe_exactly_once(backend):
    info = RHEALPixInfo(level=1)
    polygons = info.cell_boundaries(burin.full_domain(1), backend=backend)
    if backend == "geoarrow":
        polygons = shapely.polygons([shapely.linearrings(poly[0]) for poly in polygons.to_pylist()])
    assert shapely.is_valid(polygons).all()
    union = shapely.area(shapely.union_all(polygons))
    assert abs(union - 64800.0) < 1e-6
    assert abs(shapely.area(polygons).sum() - union) < 1e-6, "cells overlap"
