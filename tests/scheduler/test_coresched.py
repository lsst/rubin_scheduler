import unittest
import warnings

import healpy as hp
import numpy as np
import pandas as pd

import rubin_scheduler.scheduler.basis_functions as basis_functions
import rubin_scheduler.scheduler.detailers as dets
import rubin_scheduler.scheduler.surveys as surveys
from rubin_scheduler.scheduler.example import simple_greedy_survey
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import BaseQueueManager, CoreScheduler
from rubin_scheduler.scheduler.utils import ObservationArray, TargetoO


class DummySurvey(surveys.BaseSurvey):
    def __init__(self):
        super().__init__([], [])

    def generate_observations(self, *args, **kwargs):
        result = ObservationArray(n=10)
        result["band"] = "r"
        return result


class TestCoreSched(unittest.TestCase):
    def testsched(self):
        # Just set up a very simple survey, one band
        survey = simple_greedy_survey(bandname="r")

        qm = BaseQueueManager(detailers=[dets.RotspUpdateDetailer()])
        scheduler = CoreScheduler([survey], queue_manager=qm)

        observatory = ModelObservatory()

        # Check that we can update conditions
        scheduler.update_conditions(observatory.return_conditions())

        # Check that we can get an observation out
        obs = scheduler.request_observation()
        assert obs is not None

        # check that we can pull the whole queue
        obs_list = scheduler.request_observation(whole_queue=True)
        assert obs_list is not None

        # Check that we can flush the Queue
        scheduler.flush_queue()
        assert np.size(scheduler.queue_manager.desired_observations_array) == 0

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

        qm = BaseQueueManager(detailers=[dets.RotspUpdateDetailer()])
        scheduler = CoreScheduler(surveys, keep_rewards=True, queue_manager=qm)
        observatory = ModelObservatory()
        scheduler.update_conditions(observatory.return_conditions())

        with warnings.catch_warnings(record=True) as caught_warnings:
            scheduler.request_observation()
            for caught_warning in caught_warnings:
                if issubclass(caught_warning.category, RuntimeWarning):
                    assert str(caught_warning.message) != "All-NaN slice encountered"

        self.assertIsInstance(scheduler.queue_reward_df, pd.DataFrame)

    def test_add_obs(self):
        """Test that add_observation works with slice or array"""

        bfs = [basis_functions.ForceDelayBasisFunction(days_delay=3.0, scheduler_note="survey")]
        survey = surveys.GreedySurvey(bfs, [0.0])
        qm = BaseQueueManager(detailers=[dets.RotspUpdateDetailer()])
        scheduler = CoreScheduler([survey], queue_manager=qm)

        obs = ObservationArray()
        obs["scheduler_note"] = "survey"
        obs["mjd"] = 100
        obs["target_id"] = 1

        scheduler.add_observation(obs)

        recorded = scheduler.survey_lists[0][0].basis_functions[0].survey_features["last_obs_self"].feature

        assert recorded["mjd"] == 100
        assert recorded["scheduler_note"] == "survey"

        bfs = [basis_functions.ForceDelayBasisFunction(days_delay=3.0, scheduler_note="survey")]
        survey = surveys.GreedySurvey(bfs, [0.0])
        qm = BaseQueueManager(detailers=[dets.RotspUpdateDetailer()])
        scheduler = CoreScheduler([survey], queue_manager=qm)

        # Again, but now add just the row rather than full array
        scheduler.add_observation(obs[0])
        recorded = scheduler.survey_lists[0][0].basis_functions[0].survey_features["last_obs_self"].feature

        assert recorded["mjd"] == 100
        assert recorded["scheduler_note"] == "survey"

    def test_too_flush(self):
        """Test that the queue will flush if a new ToO comes in"""
        observatory = ModelObservatory()

        conditions = observatory.return_conditions()
        survey = DummySurvey()
        sched = CoreScheduler([survey], flush_for_new_too=True)
        sched.update_conditions(conditions)
        sched._fill_queue()

        # Make sure we have things in the queue
        assert len(sched.queue_manager.desired_observations_array) > 0

        new_too_event = TargetoO(1, np.ones(hp.nside2npix(conditions.nside)), conditions.mjd, 2)
        conditions.targets_of_opportunity = [new_too_event]

        sched.update_conditions(conditions)
        # New ToO should result in queue being cleared
        assert len(sched.queue_manager.desired_observations_array) == 0

        # Check that sending in a same ID ToO doesn't result in a new flush
        sched._fill_queue()
        # Check there is something in the queue
        assert len(sched.queue_manager.desired_observations_array) > 0
        # Update conditions. There is a ToO in the list, but
        # scheduler should recognize it's already seen it and not flush.
        sched.update_conditions(conditions)
        assert len(sched.queue_manager.desired_observations_array) > 0

        # Now make a ToO with a new ID
        new_too_event = TargetoO(10, np.ones(hp.nside2npix(conditions.nside)), conditions.mjd, 2)
        conditions.targets_of_opportunity = [new_too_event]
        # Make sure we have something in the queue
        assert len(sched.queue_manager.desired_observations_array) > 0
        sched.update_conditions(conditions)
        # Sending that in should have cleared the queue
        assert len(sched.queue_manager.desired_observations_array) == 0

        # Set the ToO to not interrupt
        new_too_event = TargetoO(
            12, np.ones(hp.nside2npix(conditions.nside)), conditions.mjd, 2, interrupt_queue=False
        )
        conditions.targets_of_opportunity = [new_too_event]
        sched._fill_queue()
        # Check there is something in the queue
        assert len(sched.queue_manager.desired_observations_array) > 0
        # Add ToOs that should not interrupt
        sched.update_conditions(conditions)
        assert len(sched.queue_manager.desired_observations_array) > 0

        # Event that wants to interrupt, but has all zero footprint, so
        # should not flush
        new_too_event = TargetoO(
            13, np.zeros(hp.nside2npix(conditions.nside)), conditions.mjd, 2, interrupt_queue=True
        )
        conditions.targets_of_opportunity = [new_too_event]
        sched.update_conditions(conditions)
        assert len(sched.queue_manager.desired_observations_array) > 0

        # Again, but set to not flush on new ToO
        conditions = observatory.return_conditions()
        survey = DummySurvey()
        sched = CoreScheduler([survey], flush_for_new_too=False)
        sched.update_conditions(conditions)
        sched._fill_queue()

        # Make sure we have something in the queue
        assert len(sched.queue_manager.desired_observations_array) > 0

        new_too_event = TargetoO(1, np.ones(hp.nside2npix(conditions.nside)), conditions.mjd, 2)
        conditions.targets_of_opportunity = [new_too_event]

        sched.update_conditions(conditions)
        # We sent in a new ToO, but should not have flushed.
        assert len(sched.queue_manager.desired_observations_array) > 0


if __name__ == "__main__":
    unittest.main()
