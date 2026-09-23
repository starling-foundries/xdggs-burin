"""A light cross-check against the reference implementation; burin-core's own fixtures are the
authoritative conformance suite."""
import numpy as np
import pytest

from xdggs_burin import RHEALPixInfo

rhealpixdggs = pytest.importorskip("rhealpixdggs")


@pytest.fixture(scope="module")
def reference():
    from rhealpixdggs.dggs import RHEALPixDGGS
    from rhealpixdggs.ellipsoids import Ellipsoid

    return RHEALPixDGGS(ellipsoid=Ellipsoid(a=6378137.0, f=1 / 298.257223563, lon_0=50.0), N_side=3)


def test_point_lookup_matches_the_reference(reference):
    import burin

    rng = np.random.default_rng(5)
    lon, lat = rng.uniform(-180, 180, 300), np.degrees(np.arcsin(rng.uniform(-1, 1, 300)))
    for level in (0, 3, 7):
        got = [burin.cid_to_suid(int(c)) for c in RHEALPixInfo(level=level).geographic2cell_ids(lon, lat)]
        want = reference.cells_from_points(lon, lat, level, plane=False).tolist()
        assert got == want


def test_centres_match_the_reference(reference):
    import burin

    cells = burin.full_domain(2)
    lon, lat = RHEALPixInfo(level=2).cell_ids2geographic(cells)
    for c, x, y in zip(cells, lon, lat):
        want = reference.cell(list(burin.cid_to_suid(int(c))[0]) + [int(d) for d in burin.cid_to_suid(int(c))[1:]]).nucleus(plane=False)
        np.testing.assert_allclose([x, y], [float(want[0]), float(want[1])], atol=1e-9)
