import unittest
import warnings

import numpy as np
import pandas as pd

import rubin_scheduler.scheduler.basis_functions as basis_functions
import rubin_scheduler.scheduler.surveys as surveys
from rubin_scheduler.scheduler.example import simple_greedy_survey
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.scheduler.utils import ObservationArray


class ScpSurvey(surveys.BaseSurvey):
    """A South Celestial Pole survey

    Simple dummy survey that should always return something"""

    def __init__(self, band, scheduler_note="scp"):
        super().__init__([])
        self.band = band
        self.scheduler_note = scheduler_note

    def generate_observations_rough(self, conditions):
        obs = ObservationArray()
        obs["RA"] = 0.0
        obs["dec"] = np.radians(-87.0)
        obs["exptime"] = 30.0  # Seconds
        obs["nexp"] = 2
        obs["band"] = self.band
        obs["scheduler_note"] = self.scheduler_note
        # Make it 3
        result = np.concatenate([obs] * 3)
        return result


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

    def test_record_rewards(self):
        # Create two surveys, so we can have one of the with a basis function
        # with all nans but stiff have the scheduler find observations it can
        # return.
        surveys = [simple_greedy_survey(bandname="r"), simple_greedy_survey(bandname="i")]

        # Modify the 1st survey so that it has an all-nan array basis function.
        new_basis_function = basis_functions.SimpleArrayBasisFunction(np.nan)
        surveys[0].basis_functions.append(new_basis_function)
        surveys[0].basis_weights.append(1)

        scheduler = CoreScheduler(surveys, keep_rewards=True)
        observatory = ModelObservatory()
        scheduler.update_conditions(observatory.return_conditions())

        with warnings.catch_warnings(record=True) as caught_warnings:
            scheduler.request_observation()
            for caught_warning in caught_warnings:
                if issubclass(caught_warning.category, RuntimeWarning):
                    assert str(caught_warning.message) != "All-NaN slice encountered"

        self.assertIsInstance(scheduler.queue_reward_df, pd.DataFrame)

    def test_condition_commands(self):
        """Test that we can send commands through Conditions object"""
        survey1 = ScpSurvey("r", scheduler_note="survey1")
        survey2 = ScpSurvey("z", scheduler_note="survey2")
        survey3 = ScpSurvey("i", scheduler_note="survey3")
        survey4 = ScpSurvey("y", scheduler_note="survey4")

        scheduler = CoreScheduler([[survey1, survey2], [survey3, survey4]])

        conditions = Conditions()
        conditions.mjd = 100
        scheduler.update_conditions(conditions)

        obs_regular = scheduler.request_observation()

        assert obs_regular["scheduler_note"] == "survey1"

        scheduler.flush_queue()

        # Disable the top tier
        conditions.commands_for_corescheduler = {"deactivate_tier": {"tier_int": 0}}
        scheduler.update_conditions(conditions)

        obs = scheduler.request_observation()
        assert obs["scheduler_note"] == "survey3"

        scheduler.flush_queue()
        scheduler.activate_tier(0)

        # Use condtions to deactivate a single survey

        conditions.commands_for_corescheduler = {"deactivate_survey": {"tier_int": 0, "survey_int": 0}}
        scheduler.update_conditions(conditions)

        obs = scheduler.request_observation()
        assert obs["scheduler_note"] == "survey2"

        # Flush, reactivate, and confirm we are back to normal

        scheduler.flush_queue()
        scheduler.activate_survey(0, 0)

        obs = scheduler.request_observation()
        assert obs["scheduler_note"] == "survey1"

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
