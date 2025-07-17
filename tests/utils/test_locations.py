import unittest

from astropy.coordinates import SkyCoord

from rubin_scheduler.utils import (
    ddf_locations,
    ddf_locations_pre3_5,
    special_locations,
)


class LocationsTests(unittest.TestCase):

    def test_ddf(self):
        old_locations = ddf_locations_pre3_5()

        for key in old_locations:
            assert isinstance(old_locations[key][0], float)
            assert isinstance(old_locations[key][1], float)

        ddf_loc = ddf_locations()
        for key in ddf_loc:
            assert isinstance(ddf_loc[key][0], float)
            assert isinstance(ddf_loc[key][1], float)

        ddf_skycoords = ddf_locations(skycoords=True)
        for key in ddf_skycoords:
            assert isinstance(ddf_skycoords[key], SkyCoord)

    def test_locations(self):

        locations = special_locations()
        for key in locations:
            assert isinstance(locations[key][0], float)
            assert isinstance(locations[key][1], float)

        locations = special_locations(skycoords=True)
        for key in locations:
            assert isinstance(locations[key], SkyCoord)


if __name__ == "__main__":
    unittest.main()
