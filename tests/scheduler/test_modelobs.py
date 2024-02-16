import unittest

import numpy as np

import rubin_scheduler.utils as utils
from rubin_scheduler.scheduler.model_observatory import ModelObservatory, acc_time, jerk_time


class KindaClouds:
    """Dummy class that always sets the clouds level to a unique float"""

    def __call__(self, mjd):
        return 0.23659


class ArbSeeing:
    """Dummy class to always return a specific seeing value"""

    def __call__(self, mjd):
        fwhm_500 = 1.756978
        return fwhm_500


class TestModelObservatory(unittest.TestCase):
    def test_ideal(self):
        """test that we can set ideal conditions"""
        mjd_start = utils.survey_start_mjd()
        mo_default = ModelObservatory(mjd_start=mjd_start)

        mo_ideal = ModelObservatory(
            mjd_start=mjd_start, cloud_data="ideal", seeing_data="ideal", downtimes="ideal"
        )

        cond_default = mo_default.return_conditions()
        cond_ideal = mo_ideal.return_conditions()

        # Should be at ideal seeing
        assert mo_ideal.fwhm_500 == 0.7
        assert mo_default.fwhm_500 != 0.7

        assert cond_ideal.bulk_cloud == 0
        assert cond_default.bulk_cloud > 0

        mjd_down = mo_default.downtimes["start"][0] + 0.01

        assert ~mo_default.check_up(mjd_down)
        assert mo_ideal.check_up(mjd_down)

    def test_replace(self):
        """test that we can replace default downtimes, seeing, and clouds"""

        mjd_start = utils.survey_start_mjd()
        mo_default = ModelObservatory(mjd_start=mjd_start)
        # Never load too many nights of sky
        mo_default.sky_model.load_length = 10.0
        cond_default = mo_default.return_conditions()

        # Define new downtimes
        downtimes = np.zeros(2, dtype=list(zip(["start", "end"], [float, float])))
        downtimes["start"] = np.array([1, 10]) + mjd_start
        downtimes["end"] = np.array([2, 11]) + mjd_start

        seeing_data = ArbSeeing()
        cloud_data = KindaClouds()

        mo_new = ModelObservatory(
            mjd_start=mjd_start,
            seeing_data=seeing_data,
            cloud_data=cloud_data,
            downtimes=downtimes,
        )
        # Never load too many nights of sky
        mo_new.sky_model.load_length = 10.0
        cond_new = mo_new.return_conditions()

        # Make sure the internal downtimes are different
        assert ~np.array_equal(mo_default.downtimes, mo_new.downtimes)

        # Make sure seeing is not the same
        diff = cond_default.fwhm_eff["r"] - cond_new.fwhm_eff["r"]
        assert np.nanmin(np.abs(diff)) > 0

        # Make sure cloudyness level is not the same
        assert cond_default.bulk_cloud != cond_new.bulk_cloud

    def test_jerk(self):
        """Test that motion times using jerk are reasonable"""

        distance = 4.5  # degrees
        v_max = 1.75  # deg/s
        acc_max = 0.875  # deg/s/s
        jerk_max = 5.0  # deg/s/s/s

        # Test that adding jerk increases travel time
        t1 = jerk_time(distance, v_max, acc_max, jerk_max)
        t2 = acc_time(distance, v_max, acc_max)

        assert t1 > t2

        # Test that jerk of None reverts properly
        t1 = jerk_time(distance, v_max, acc_max, None)
        t2 = acc_time(distance, v_max, acc_max)

        assert t1 == t2

        # Test that large jerk is close to acceleration only time

        t1 = jerk_time(distance, v_max, acc_max, 1e9)
        t2 = acc_time(distance, v_max, acc_max)

        assert np.allclose(t1, t2, atol=1e-5)

        # Test that degrees or radians are the same
        t1 = jerk_time(distance, v_max, acc_max, jerk_max)
        t2 = jerk_time(np.radians(distance), np.radians(v_max), np.radians(acc_max), np.radians(jerk_max))

        assert np.allclose(t1, t2, atol=1e-7)

        # test if values are equal case.
        distance = 4.5  # degrees
        v_max = 3.5  # deg/s
        acc_max = 3.5  # deg/s/s
        jerk_max = 3.5  # deg/s/s/s

        t1 = jerk_time(distance, v_max, acc_max, jerk_max)


if __name__ == "__main__":
    unittest.main()
