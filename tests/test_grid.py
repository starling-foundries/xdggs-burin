import json

import burin
import numpy as np
import pytest
import shapely
import shapely.testing

from xdggs_burin import GRID_NAME, RHEALPixInfo

Q453 = burin.suid_to_cid("Q453")


def geoarrow_to_shapely(arr):
    return shapely.polygons([shapely.linearrings(poly[0]) for poly in arr.to_pylist()])


class TestRHEALPixInfo:
    @pytest.mark.parametrize("level", [0, 1, 5, 15])
    def test_init(self, level):
        assert RHEALPixInfo(level=level).level == level

    @pytest.mark.parametrize("level", [-1, 16])
    def test_init_invalid_level(self, level):
        with pytest.raises(ValueError, match="level must be an integer between 0 and 15"):
            RHEALPixInfo(level=level)

    def test_init_other_ellipsoid(self):
        with pytest.raises(ValueError, match="WGS84"):
            RHEALPixInfo(level=3, ellipsoid="sphere")

    @pytest.mark.parametrize(
        "mapping",
        [
            {"grid_name": GRID_NAME, "level": 4},
            {"grid_name": GRID_NAME, "resolution": 4},
            {"grid_name": GRID_NAME, "refinement_level": np.int64(4)},
            {"grid_name": GRID_NAME, "level": 4, "indexing_scheme": "cid"},
            {"grid_name": GRID_NAME, "level": 4, "lon_0": 50, "north_square": 0, "south_square": 0, "ellipsoid": "WGS84"},
            {"grid_name": GRID_NAME, "level": 4, "ellipsoid": "wgs84"},
        ],
    )
    def test_from_dict(self, mapping):
        info = RHEALPixInfo.from_dict(mapping)
        assert info.level == 4 and type(info.level) is int

    @pytest.mark.parametrize(
        "extra",
        [{"lon_0": 0}, {"north_square": 1}, {"south_square": 2}, {"ellipsoid": "sphere"}, {"indexing_scheme": "suid"}],
    )
    def test_from_dict_refuses_another_grid(self, extra):
        with pytest.raises(ValueError):
            RHEALPixInfo.from_dict({"level": 4} | extra)

    def test_from_dict_refuses_two_levels(self):
        with pytest.raises(ExceptionGroup):
            RHEALPixInfo.from_dict({"level": 4, "resolution": 5})

    def test_roundtrip(self):
        mapping = {"grid_name": GRID_NAME, "level": 7, "indexing_scheme": "cid"}
        assert RHEALPixInfo.from_dict(mapping).to_dict() == mapping

    def test_cell_ids2geographic(self):
        lon, lat = RHEALPixInfo(level=3).cell_ids2geographic(np.array([Q453]))
        expected = burin.cells_to_lonlat([Q453])
        np.testing.assert_array_equal(lon, expected[0])
        np.testing.assert_array_equal(lat, expected[1])

    def test_cell_ids2geographic_wrong_level(self):
        with pytest.raises(ValueError, match="not a cell at level 4"):
            RHEALPixInfo(level=4).cell_ids2geographic(np.array([Q453]))

    def test_geographic2cell_ids(self):
        info = RHEALPixInfo(level=3)
        lon, lat = info.cell_ids2geographic(np.array([Q453]))
        actual = info.geographic2cell_ids(lon=lon, lat=lat)
        assert actual.dtype == np.uint64
        np.testing.assert_array_equal(actual, [Q453])

    def test_geographic2cell_ids_accepts_int64_input_ids(self):
        info = RHEALPixInfo(level=3)
        lon, lat = info.cell_ids2geographic(np.array([Q453], dtype="int64"))
        np.testing.assert_array_equal(info.geographic2cell_ids(lon, lat), [Q453])

    @pytest.mark.parametrize("backend", ["shapely", "geoarrow"])
    def test_cell_boundaries(self, backend):
        info = RHEALPixInfo(level=2)
        cells = burin.full_domain(2)
        polygons = info.cell_boundaries(cells, backend=backend)
        if backend == "geoarrow":
            meta = polygons.field.metadata
            assert meta[b"ARROW:extension:name"] == b"geoarrow.polygon"
            assert json.loads(meta[b"ARROW:extension:metadata"])["crs"]["id"]["code"] == 4326
            polygons = geoarrow_to_shapely(polygons)
        assert len(polygons) == cells.size
        assert shapely.is_valid(polygons).all()
        assert shapely.is_ccw(shapely.get_exterior_ring(polygons)).all()
        expected = info.cell_boundaries(cells, backend="shapely")
        shapely.testing.assert_geometries_equal(polygons, expected)

    def test_cell_boundaries_invalid_backend(self):
        with pytest.raises(ValueError, match="invalid backend"):
            RHEALPixInfo(level=1).cell_boundaries([burin.suid_to_cid("Q4")], backend="wkt")

    def test_cell_boundaries_hold_their_centres(self):
        info = RHEALPixInfo(level=3)
        cells = burin.full_domain(3)
        polygons = info.cell_boundaries(cells)
        lon, lat = info.cell_ids2geographic(cells)
        shifted = np.where((shapely.bounds(polygons)[:, 2] > 180) & (lon < 0), lon + 360, lon)
        caps = np.array([burin.cid_to_suid(int(c))[0] in "NS" and set(burin.cid_to_suid(int(c))[1:]) <= {"4"} for c in cells])
        assert caps.sum() == 2
        inside = shapely.contains_xy(polygons, shifted, lat)
        assert inside[~caps].all()
        assert (shapely.bounds(polygons[caps])[:, 1] == -90.0).sum() == 1
        assert (shapely.bounds(polygons[caps])[:, 3] == 90.0).sum() == 1

    def test_polar_caps_reach_the_pole(self):
        info = RHEALPixInfo(level=2)
        north, south = burin.suid_to_cid("N44"), burin.suid_to_cid("S44")
        polygons = info.cell_boundaries([north, south])
        assert shapely.bounds(polygons[0])[3] == 90.0
        assert shapely.bounds(polygons[1])[1] == -90.0

    def test_antimeridian_cells_are_whole(self):
        info = RHEALPixInfo(level=5)
        cells = info.geographic2cell_ids(np.array([179.99, -179.99]), np.array([0.0, 0.0]))
        polygons = info.cell_boundaries(cells)
        widths = shapely.bounds(polygons)[:, 2] - shapely.bounds(polygons)[:, 0]
        assert (widths < 1.0).all(), "a crossing cell is drawn narrow, not around the world"

    def test_zoom_to_parents(self):
        actual = RHEALPixInfo(level=3).zoom_to(np.array([Q453]), level=1)
        np.testing.assert_array_equal(actual, [burin.suid_to_cid("Q4")])

    def test_zoom_to_children(self):
        actual = RHEALPixInfo(level=1).zoom_to(np.array([burin.suid_to_cid("Q4")]), level=2)
        assert actual.shape == (1, 9)
        assert [burin.cid_to_suid(int(c)) for c in actual[0]] == [f"Q4{k}" for k in range(9)]
