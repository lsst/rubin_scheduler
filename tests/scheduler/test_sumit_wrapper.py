import unittest

import numpy as np

import rubin_scheduler.utils as utils
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import SummitWrapper


class TestSummitWrapper(unittest.TestCase):

    def test_baseline(self):

        # Run the usual baseline
        mjd_start = utils.SURVEY_START_MJD
        nside = 32
        survey_length = 2.0  # days
        n_visit_limit = 200

        # Reset and run with a wrapped CoreScheduler
        scheduler_wrapped = SummitWrapper(example_scheduler(nside=nside, survey_start_mjd=mjd_start))
        observatory = ModelObservatory(
            nside=nside, mjd_start=mjd_start, cloud_data="ideal", downtimes="ideal"
        )
        observatory, scheduler_wrapped, observations_wrapped = sim_runner(
            observatory,
            scheduler_wrapped,
            sim_duration=survey_length,
            filename=None,
            n_visit_limit=n_visit_limit,
        )

        scheduler = example_scheduler(nside=nside, survey_start_mjd=mjd_start)
        observatory = ModelObservatory(
            nside=nside, mjd_start=mjd_start, cloud_data="ideal", downtimes="ideal"
        )
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, sim_duration=survey_length, filename=None, n_visit_limit=n_visit_limit
        )

        # Should be the same number of unique target IDs
        assert np.size(np.unique(observations["target_id"])) == np.size(
            np.unique(observations_wrapped["target_id"])
        )

        # I don't mind if target_id got ahead on one method
        observations_wrapped["target_id"] = observations["target_id"]

        # Should be the same
        np.testing.assert_array_equal(observations, observations_wrapped)


if __name__ == "__main__":
    unittest.main()
