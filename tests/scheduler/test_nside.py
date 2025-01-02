import os
import unittest

import numpy as np

import rubin_scheduler.utils as utils
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.basis_functions import SunAltLimitBasisFunction
from rubin_scheduler.scheduler.example import simple_greedy_survey, simple_pairs_survey
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler

SAMPLE_BIG_DATA_FILE = os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")


class TestNside(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_nside(self):
        """
        test running at higher nside
        """
        mjd_start = utils.SURVEY_START_MJD
        nside = 64
        observatory = ModelObservatory(
            nside=nside, mjd_start=mjd_start, cloud_data="ideal", downtimes="ideal"
        )
        conditions = observatory.return_conditions()
        # set the mjd to start of night
        observatory.mjd = conditions.sun_n12_setting + 1.01
        survey_length = 0.1  # days

        # Add an avoidance of twilight+ for the pairs surveys -
        # this ensures greedy survey will have some time to operate
        pairs_surveys = [
            simple_pairs_survey(bandname="g", bandname2="r", nside=nside),
            simple_pairs_survey(bandname="i", bandname2="z", nside=nside),
        ]
        for survey in pairs_surveys:
            survey.basis_functions.append(SunAltLimitBasisFunction(alt_limit=-22))
            survey.basis_weights.append(0)
        greedy_surveys = [
            simple_greedy_survey(bandname="z", nside=nside),
        ]

        scheduler = CoreScheduler([pairs_surveys, greedy_surveys], nside=nside)

        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, sim_duration=survey_length, filename=None
        )

        # Make sure some blobs executed
        assert "simple pair 30, iz, a" in observations["scheduler_note"]
        assert "simple pair 30, iz, b" in observations["scheduler_note"]
        # Make sure some greedy executed
        assert "simple greedy z" in observations["scheduler_note"]
        # Make sure lots of observations executed
        assert observations.size > 50
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0


if __name__ == "__main__":
    unittest.main()
