import burin
import numpy as np
import pytest

from xdggs_burin import RHEALPixIndex, RHEALPixInfo, from_dggs_json

AOI = {"type": "Polygon", "coordinates": [[[-0.13, 51.50], [-0.10, 51.50], [-0.10, 51.52], [-0.13, 51.52], [-0.13, 51.50]]]}


def _index(cells, level):
    return RHEALPixIndex(np.asarray(cells, dtype=np.uint64), "cells", "cell_ids", RHEALPixInfo(level=level))


def test_fingerprint_ignores_order_and_duplicates():
    cells = np.array(burin.polyfill(AOI, 10), dtype=np.uint64)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(np.concatenate([cells, cells[:5]]))
    assert _index(cells, 10).fingerprint() == _index(shuffled, 10).fingerprint()


def test_fingerprint_is_the_polygon_fingerprint():
    cells = burin.polyfill(AOI, 10)
    assert _index(cells, 10).fingerprint() == burin.fingerprint_polygon(AOI, 10)


def test_fingerprint_depends_on_level_and_cells():
    cells = np.array(burin.polyfill(AOI, 9), dtype=np.uint64)
    finer = burin.zoom_to(cells, 9, 10).ravel()
    assert _index(cells, 9).fingerprint() != _index(finer, 10).fingerprint()
    assert _index(cells, 9).fingerprint() != _index(cells[1:], 9).fingerprint()


def test_dggs_json_round_trip():
    cells = np.array(burin.polyfill(AOI, 10), dtype=np.uint64)
    index = _index(cells, 10)
    doc = index.to_dggs_json()
    assert doc["dggrs"] == "https://www.opengis.net/def/dggrs/OGC/1.0/rHEALPix"
    back = RHEALPixIndex.from_dggs_json(doc)
    assert back.grid_info.level == 10
    assert back.fingerprint() == index.fingerprint()
    ds = from_dggs_json(doc)
    assert ds.dggs.index.fingerprint() == index.fingerprint()


def test_dggs_json_refusals():
    q453, r0 = burin.suid_to_cid("Q453"), burin.suid_to_cid("R000")
    with pytest.raises(ValueError, match="more than one base cell"):
        _index([q453, r0], 3).to_dggs_json()
    with pytest.raises(ValueError, match="not every cell lies in zone"):
        _index([q453, r0], 3).to_dggs_json(zone=burin.suid_to_cid("Q4"))
    with pytest.raises(ValueError, match="levels below the zone"):
        _index(np.array([q453 * 9**10, q453 * 9**10 + 1], dtype=np.uint64), 13).to_dggs_json(zone=burin.suid_to_cid("Q"))
    with pytest.raises(ValueError, match="no cells"):
        _index([], 3).to_dggs_json()
