import unittest

import numpy as np

from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import BaseQueueManager, CoreScheduler
from rubin_scheduler.scheduler.surveys import BaseSurvey
from rubin_scheduler.scheduler.utils import ObservationArray
from rubin_scheduler.utils import (
    _approx_altaz2pa,
    _approx_ra_dec2_alt_az,
    rotation_converter,
)


class PoleSurvey(BaseSurvey):
    """Simple survey that points at the pole"""

    def __init__(self, rot_key="rotSkyPos", rot_value=0.0):
        self.rot_key = rot_key
        self.rot_value = np.radians(rot_value)
        super().__init__([], [])

    def generate_observations(self, conditions):

        result = ObservationArray(1)

        result["rotSkyPos"] = np.nan
        result["rotTelPos"] = np.nan

        result["RA"] = 0
        result["dec"] = -np.pi / 2
        result["band"] = "r"
        result[self.rot_key] = self.rot_value
        return result


class TestRotator(unittest.TestCase):

    def test_rotator(self):
        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        rc = rotation_converter()

        alt, az = _approx_ra_dec2_alt_az(
            0,
            -np.pi / 2.0,
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
        rsp = rc.rottelpos2rotskypos(20.0, obs_pa)

        # Pass in a rotSkyPos. Should just go through
        s1 = PoleSurvey(rot_key="rotSkyPos", rot_value=rsp)

        sched = CoreScheduler([s1])
        sched.update_conditions(conditions)

        out_obs = sched.request_observation(conditions)
        assert np.isfinite(out_obs["rotSkyPos"])
        assert not np.isfinite(out_obs["rotTelPos"])

        # Set rotSkyPos to NaN, set rotTelPos to 20 degrees.
        # QueueManager should update rotSkyPos to a valid value
        s2 = PoleSurvey(rot_key="rotTelPos", rot_value=20.0)
        sched = CoreScheduler([s2])
        sched.update_conditions(conditions)

        out_obs = sched.request_observation(conditions)
        assert np.isfinite(out_obs["rotSkyPos"])

        # Test that we can force a rotTelPos value out
        # Have to force QueueManager to do nothing
        qm = BaseQueueManager(detailers=[])
        s3 = PoleSurvey(rot_key="rotTelPos", rot_value=20.0)
        sched = CoreScheduler([s3], queue_manager=qm)
        sched.update_conditions(conditions)

        out_obs = sched.request_observation(conditions)

        assert ~np.isfinite(out_obs["rotSkyPos"])
        assert np.isfinite(out_obs["rotTelPos"])


if __name__ == "__main__":
    unittest.main()
