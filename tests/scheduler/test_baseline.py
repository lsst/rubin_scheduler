import os
import unittest

import numpy as np

import rubin_scheduler.utils as utils
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.basis_functions import SunAltLimitBasisFunction
from rubin_scheduler.scheduler.example import example_scheduler, simple_greedy_survey, simple_pairs_survey
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
        _conditions = super().return_conditions()

        # Always have a strong wind from the north
        wind_speed = 40.0
        wind_direction = 0.0
        self.conditions.wind_speed = wind_speed
        self.conditions.wind_direction = wind_direction

        return self.conditions


class TestExample(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_example(self):
        """Try out the example scheduler."""
        mjd_start = utils.SURVEY_START_MJD
        nside = 32
        survey_length = 4.0  # days
        scheduler = example_scheduler(nside=nside, mjd_start=mjd_start)
        observatory = ModelObservatory(nside=nside, mjd_start=mjd_start)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, sim_duration=survey_length, filename=None
        )
        # Check that greedy observed some
        assert "greedy" in observations["scheduler_note"]
        # check some long pairs got observed
        assert np.any(["pair_33" in obs for obs in observations["scheduler_note"]])
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0
        # Make sure a DDF executed
        assert np.any(["DD" in note for note in observations["scheduler_note"]])

    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_wind(self):
        """
        Test that a wind mask prevent things from being executed in
        the wrong spot
        """
        mjd_start = utils.SURVEY_START_MJD
        nside = 32
        survey_length = 2.0  # days

        surveys = [simple_greedy_survey(filtername=f) for f in "gri"]

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatoryWindy(
            nside=nside, mjd_start=mjd_start, downtimes="ideal", cloud_data="ideal"
        )
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, sim_duration=survey_length, filename=None
        )

        # Make sure lots of observations executed, but allow short night
        # if survey_start changes
        assert observations.size > 700 * survey_length
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0

        # Make sure nothing executed in the strong wind
        assert np.min(np.degrees(observations["az"])) > 30.0
        assert np.max(np.degrees(observations["az"])) < (360.0 - 30.0)

    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_nside(self):
        """
        test running at higher nside
        """
        mjd_start = utils.SURVEY_START_MJD
        nside = 64
        survey_length = 2.0  # days

        # Add an avoidance of twilight+ for the pairs surveys -
        # this ensures greedy survey will have some time to operate
        pairs_surveys = [
            simple_pairs_survey(filtername="g", filtername2="r", nside=nside),
            simple_pairs_survey(filtername="i", filtername2="z", nside=nside),
        ]
        for survey in pairs_surveys:
            survey.basis_functions.append(SunAltLimitBasisFunction(alt_limit=-22))
            survey.basis_weights.append(0)
        greedy_surveys = [
            simple_greedy_survey(filtername="z", nside=nside),
        ]

        scheduler = CoreScheduler([pairs_surveys, greedy_surveys], nside=nside)
        observatory = ModelObservatory(nside=nside, mjd_start=mjd_start)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, sim_duration=survey_length, filename=None
        )

        # Make sure some blobs executed
        assert "simple pair 30, iz, a" in observations["scheduler_note"]
        assert "simple pair 30, iz, b" in observations["scheduler_note"]
        # Make sure some greedy executed
        assert "simple greedy z" in observations["scheduler_note"]
        # Make sure lots of observations executed
        assert observations.size > 800
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0


if __name__ == "__main__":
    unittest.main()
