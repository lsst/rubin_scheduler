import unittest

import pandas as pd

import rubin_scheduler.scheduler.basis_functions as basis_functions
import rubin_scheduler.scheduler.surveys as surveys
from rubin_scheduler.scheduler.example import simple_greedy_survey
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.scheduler.utils import ObservationArray


class TestCoreSched(unittest.TestCase):
    def testsched(self):
        # Just set up a very simple survey, one band
        survey = simple_greedy_survey(bandname="r")

        scheduler = CoreScheduler([survey])

        observatory = ModelObservatory()

        # Check that we can update conditions
        scheduler.update_conditions(observatory.return_conditions())

        # Check that we can get an observation out
        obs = scheduler.request_observation()
        assert obs is not None

        # check that we can pull the whole queue
        obs_list = scheduler.request_observation(whole_queue=True)
        assert obs_list is not None
        assert len(scheduler.queue) == 0

        # Check that we can flush the Queue
        scheduler.flush_queue()
        assert len(scheduler.queue) == 0

        # Check that we can add an observation
        scheduler.add_observation(obs)

        # Check dunder methods
        self.assertIsInstance(repr(scheduler), str)
        self.assertIsInstance(str(scheduler), str)

        # Check access methods
        _ = scheduler.get_basis_functions([0, 0])
        _ = scheduler.get_healpix_maps([0, 0])

        # Check survey access methods
        reward_df = scheduler.make_reward_df(observatory.return_conditions())
        self.assertIsInstance(reward_df, pd.DataFrame)
        reward_df = scheduler.make_reward_df(observatory.return_conditions(), accum=False)
        self.assertIsInstance(reward_df, pd.DataFrame)

        obs = scheduler.request_observation()
        surveys_df = scheduler.surveys_df(0)
        self.assertIsInstance(surveys_df, pd.DataFrame)

        # Test we can record basis function values when requested
        recording_scheduler = CoreScheduler([survey], keep_rewards=True)
        recording_scheduler.update_conditions(observatory.return_conditions())
        obs = recording_scheduler.request_observation()
        self.assertIsInstance(recording_scheduler.queue_reward_df, pd.DataFrame)

    def test_add_obs(self):
        """Test that add_observation works with slice or array"""

        bfs = [basis_functions.ForceDelayBasisFunction(days_delay=3.0, scheduler_note="survey")]
        survey = surveys.GreedySurvey(bfs, [0.0])
        scheduler = CoreScheduler([survey])

        obs = ObservationArray()
        obs["scheduler_note"] = "survey"
        obs["mjd"] = 100

        scheduler.add_observation(obs)

        recorded = scheduler.survey_lists[0][0].basis_functions[0].survey_features["last_obs_self"].feature

        assert recorded["mjd"] == 100
        assert recorded["scheduler_note"] == "survey"

        bfs = [basis_functions.ForceDelayBasisFunction(days_delay=3.0, scheduler_note="survey")]
        survey = surveys.GreedySurvey(bfs, [0.0])
        scheduler = CoreScheduler([survey])

        # Again, but now add just the row rather than full array
        scheduler.add_observation(obs[0])
        recorded = scheduler.survey_lists[0][0].basis_functions[0].survey_features["last_obs_self"].feature

        assert recorded["mjd"] == 100
        assert recorded["scheduler_note"] == "survey"


if __name__ == "__main__":
    unittest.main()
