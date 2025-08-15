import os
import unittest

import numpy as np

import rubin_scheduler.utils as utils
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory

SAMPLE_BIG_DATA_FILE = os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")


class TestExample(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_example(self):
        """Try out the example scheduler."""
        mjd_start = utils.SURVEY_START_MJD
        nside = 32
        survey_length = 4.0  # days
        scheduler = example_scheduler(nside=nside, mjd_start=mjd_start)
        observatory = ModelObservatory(
            nside=nside, mjd_start=mjd_start, cloud_data="ideal", downtimes="ideal"
        )
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
        # Make sure a twilight executed
        assert np.any(["twilight_near_sun, 1" in note for note in observations["scheduler_note"]])


if __name__ == "__main__":
    unittest.main()
