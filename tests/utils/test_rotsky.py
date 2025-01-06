import unittest

import numpy as np

from rubin_scheduler.scheduler.utils import smallest_signed_angle
from rubin_scheduler.utils import (
    SURVEY_START_MJD,
    Site,
    _approx_ra_dec2_alt_az,
    pseudo_parallactic_angle,
    rotation_converter,
)


class TestRotSkyConvert(unittest.TestCase):
    def test_rotation_converter(self):
        pa_vals = np.linspace(0, 360, 15)
        for tn in ["rubin", "auxtel", "comcam"]:
            rc = rotation_converter(telescope=tn)
            for pa in pa_vals:
                in_vals = np.linspace(-180, 180, 15)
                step1 = rc.rottelpos2rotskypos(in_vals, pa)
                step2 = rc.rotskypos2rottelpos(step1, pa)
                diff = np.abs(step2 - in_vals)
                pot_flip = np.where(diff > 1)
                diff[pot_flip] = diff[pot_flip] - 360
                assert np.allclose(diff, diff * 0)

                in_vals = np.linspace(0, 360, 15)
                step1 = rc.rotskypos2rottelpos(in_vals, pa)
                step2 = rc.rottelpos2rotskypos(step1, pa)
                diff = np.abs(step2 - in_vals)
                pot_flip = np.where(np.abs(diff) > 1)
                diff[pot_flip] = diff[pot_flip] - 360
                assert np.allclose(diff, diff * 0)
        # check that scalars work as well
        for tn in ["rubin", "auxtel", "comcam"]:
            rc = rotation_converter(telescope=tn)
            for pa in pa_vals:
                in_vals = np.linspace(-180, 180, 15)
                for iv in in_vals:
                    step1 = rc.rottelpos2rotskypos(iv, pa)
                    step2 = rc.rotskypos2rottelpos(step1, pa)
                    diff = abs(step2 - iv)
                    if diff > 1:
                        diff = diff - 360
                    assert np.allclose(diff, diff * 0)

                in_vals = np.linspace(0, 360, 15)
                for iv in in_vals:
                    step1 = rc.rotskypos2rottelpos(iv, pa)
                    step2 = rc.rottelpos2rotskypos(step1, pa)
                    diff = abs(step2 - iv)
                    if diff > 1:
                        diff = diff - 360
                    assert np.allclose(diff, diff * 0)

        with self.assertRaises(ValueError):
            rc = rotation_converter(telescope="not_a_telescope_name")

    def test_pseudo_pa(self):
        # Check that the pseudo parallactic angle is
        # somewhat close to the approx parallactic angle
        lsst = Site("LSST")
        rng = np.random.default_rng(seed=42)

        n = 100
        ra = rng.uniform(low=0, high=360, size=n)
        dec = rng.uniform(low=-90, high=90, size=n)
        mjd = np.arange(n) + SURVEY_START_MJD

        psudo_pa, salt, saz = pseudo_parallactic_angle(ra, dec, mjd, lon=lsst.longitude, height=lsst.height)

        falt, faz, fpa = _approx_ra_dec2_alt_az(
            np.radians(ra),
            np.radians(dec),
            np.radians(lsst.latitude),
            np.radians(lsst.longitude),
            mjd,
            return_pa=True,
        )

        diff = smallest_signed_angle(psudo_pa, np.degrees(fpa))
        diff[np.where(diff > 90)] -= 180
        diff[np.where(diff < -90)] += 180

        # Say they should be within 5 degrees
        assert np.max(np.abs(diff)) < 5.0


if __name__ == "__main__":
    unittest.main()
