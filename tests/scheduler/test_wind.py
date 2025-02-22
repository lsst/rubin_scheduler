import os
import unittest

import numpy as np

import rubin_scheduler.utils as utils
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import simple_greedy_survey
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler

SAMPLE_BIG_DATA_FILE = os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")


class ModelObservatoryWindy(ModelObservatory):
    """Have the model observatory always have a strong wind from the north"""

    def return_conditions(self):
        """
        Returns
        -------
        rubin_scheduler.scheduler.features.conditions object
        """
        self.conditions = super().return_conditions()

        # Always have a strong wind from the north
        wind_speed = 40.0
        wind_direction = 0.0
        self.conditions.wind_speed = wind_speed
        self.conditions.wind_direction = wind_direction

        return self.conditions


class TestWind(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_wind(self):
        """
        Test that a wind mask prevent things from being executed in
        the wrong spot
        """
        mjd_start = utils.SURVEY_START_MJD
        nside = 32
        survey_length = 1.0  # days

        surveys = [simple_greedy_survey(bandname=f) for f in "gri"]

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatoryWindy(
            nside=nside, mjd_start=mjd_start, downtimes="ideal", cloud_data="ideal"
        )
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, sim_duration=survey_length, filename=None, n_visit_limit=501
        )

        # Make sure lots of observations executed, but allow short night
        # if survey_start changes
        assert observations.size > 500
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0

        # Make sure nothing executed in the strong wind
        assert np.min(np.degrees(observations["az"])) > 30.0
        assert np.max(np.degrees(observations["az"])) < (360.0 - 30.0)


if __name__ == "__main__":
    unittest.main()
