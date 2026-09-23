import burin
import numpy as np
import pytest
import xarray as xr

from xdggs_burin import GRID_NAME, RHEALPixIndex, RHEALPixInfo

CELLS = np.array([burin.suid_to_cid(s) for s in ("Q453", "Q454", "R000", "N888")], dtype=np.uint64)


def test_init():
    index = RHEALPixIndex(CELLS, "cells", "cell_ids", RHEALPixInfo(level=3))
    assert index._grid == RHEALPixInfo(level=3)
    assert index._dim == "cells" and index._name == "cell_ids"
    assert index._index.index.name == "cell_ids"
    np.testing.assert_array_equal(index.values(), CELLS)


def test_from_variables():
    var = xr.Variable("cells", CELLS, {"grid_name": GRID_NAME, "level": 3})
    index = RHEALPixIndex.from_variables({"cell_ids": var}, options={})
    assert index.grid_info.level == 3
    np.testing.assert_array_equal(index.values(), CELLS)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_full_domain(level):
    index = RHEALPixIndex.full_domain(level)
    values = index.values()
    assert values.size == 6 * 9**level
    assert (np.diff(values.astype("int64")) == 1).all()
    assert (burin.cell_levels(values) == level).all()


def test_replace():
    old = RHEALPixIndex(CELLS, "cells", "cell_ids", RHEALPixInfo(level=3))
    new = old._replace(old._index)
    assert new._grid == old._grid and new._dim == old._dim and new._name == old._name


def test_repr_inline():
    index = RHEALPixIndex(CELLS, "cells", "cell_ids", RHEALPixInfo(level=3))
    assert index._repr_inline_(70) == "RHEALPixIndex(level=3)"


def _ds(cells, level):
    return xr.Dataset(coords={"cell_ids": ("cells", cells, {"grid_name": GRID_NAME, "level": level})}).pipe(
        lambda ds: __import__("xdggs").decode(ds)
    )


def test_join_and_reindex():
    a, b = _ds(CELLS, 3), _ds(CELLS[1:], 3)
    joined = xr.align(a, b, join="inner")[0]
    assert joined.sizes["cells"] == 3
    other = _ds(burin.zoom_to(CELLS, 3, 2), 2)
    with pytest.raises(ValueError, match="different grid parameters"):
        xr.align(a, other, join="inner")


def test_equals():
    a = RHEALPixIndex(CELLS, "cells", "cell_ids", RHEALPixInfo(level=3))
    assert a.equals(RHEALPixIndex(CELLS, "cells", "cell_ids", RHEALPixInfo(level=3)))
    assert not a.equals(RHEALPixIndex(CELLS[:2], "cells", "cell_ids", RHEALPixInfo(level=3)))
