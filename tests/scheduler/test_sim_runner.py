import logging
import unittest

import numpy as np

import rubin_scheduler.utils as utils
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.scheduler.surveys import BaseSurvey


class NoObsSurvey(BaseSurvey):
    """Dummy class that always returns no valid reward"""

    def calc_reward_function(self, conditions):
        return -np.inf


class TestSimRunner(unittest.TestCase):

    def test_no_obs(self):
        """Check that sim ends even if we stop returning observations."""
        mjd_start = utils.SURVEY_START_MJD
        nside = 32
        survey_length = 1.5  # days

        scheduler = CoreScheduler([NoObsSurvey([], detailers=[])])
        observatory = ModelObservatory(nside=nside, mjd_start=mjd_start)
        # Turn off noisy log warnings
        logging.disable(logging.CRITICAL)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, sim_duration=survey_length, filename=None
        )

        assert len(observations) == 0

        observatory, scheduler, observations, reward_df, obs_rewards_series = sim_runner(
            observatory, scheduler, sim_duration=survey_length, filename=None, record_rewards=True
        )

        assert len(observations) == 0


if __name__ == "__main__":
    unittest.main()
