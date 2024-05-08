import unittest

import numpy as np

from rubin_scheduler.utils import rotation_converter


class TestRotSkyConvert(unittest.TestCase):
    def test_rotation_converter(self):
        pa_vals = np.linspace(0, 360, 15)
        for tn in ["rubin", "auxtel"]:
            rc = rotation_converter(telescope=tn)
            for pa in pa_vals:
                in_vals = np.linspace(-180, 180, 15)
                step1 = rc.rottelpos2rotskypos(in_vals, pa)
                step2 = rc.rotskypos2rottelpos(step1, pa)
                diff = step2 - in_vals
                pot_flip = np.where(np.abs(diff) > 1)
                diff[pot_flip] = diff[pot_flip] % 360
                assert np.allclose(diff, diff * 0)

                in_vals = np.linspace(0, 360, 15)
                step1 = rc.rotskypos2rottelpos(in_vals, pa)
                step2 = rc.rottelpos2rotskypos(step1, pa)
                diff = step2 - in_vals
                pot_flip = np.where(np.abs(diff) > 1)
                diff[pot_flip] = diff[pot_flip] % 360
                assert np.allclose(diff, diff * 0)
        # check that scalars work as well
        for tn in ["rubin", "auxtel"]:
            rc = rotation_converter(telescope=tn)
            for pa in pa_vals:
                in_vals = np.linspace(-180, 180, 15)
                for iv in in_vals:
                    step1 = rc.rottelpos2rotskypos(iv, pa)
                    step2 = rc.rotskypos2rottelpos(step1, pa)
                    diff = step2 - iv
                    if abs(diff) > 1:
                        diff = diff % 360
                    assert np.allclose(diff, diff * 0)

                in_vals = np.linspace(0, 360, 15)
                for iv in in_vals:
                    step1 = rc.rotskypos2rottelpos(iv, pa)
                    step2 = rc.rottelpos2rotskypos(step1, pa)
                    diff = step2 - iv
                    if abs(diff) > 1:
                        diff = diff % 360
                    assert np.allclose(diff, diff * 0)

        with self.assertRaises(ValueError):
            rc = rotation_converter(telescope="not_a_telescope_name")


if __name__ == "__main__":
    unittest.main()
