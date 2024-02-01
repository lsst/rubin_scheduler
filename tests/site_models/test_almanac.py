import unittest

import numpy as np
import pandas as pd

from rubin_scheduler.site_models import Almanac


class TestAlmanac(unittest.TestCase):
    def test_alm(self):
        alma = Almanac()

        mjd = 59853.35

        # Dead simple make sure the things load.
        planets = alma.get_planet_positions(mjd)
        sun = alma.get_sunset_info(mjd)
        moon = alma.get_sun_moon_positions(mjd)
        indx = alma.mjd_indx(mjd)

        assert planets is not None
        assert sun is not None
        assert moon is not None
        assert indx is not None

    def test_index_for_local_evening(self):
        almanac = Almanac()
        longitude = np.radians(-70.75)
        test_dates = ["2024-12-22", "2024-03-15", "2024-06-22"]
        for test_date in test_dates:
            night_index = almanac.index_for_local_evening(test_date, longitude)
            sunset_mjd = almanac.sunsets[night_index]["sunset"]
            sunset_timestamp = (sunset_mjd - 40587) * 24 * 60 * 60
            sunset_local_iso = (
                pd.Timestamp(sunset_timestamp, tz="UTC", unit="s").tz_convert("Chile/Continental").isoformat()
            )
            self.assertTrue(sunset_local_iso.startswith(test_date))


if __name__ == "__main__":
    unittest.main()
