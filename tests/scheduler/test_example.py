import os
import unittest

import numpy as np

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.example import example_scheduler, run_sched
from rubin_scheduler.utils import survey_start_mjd


class TestExample(unittest.TestCase):

    @unittest.skipUnless(
        os.path.isfile(os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")),
        "Test data not available.",
    )
    def test_example(self):
        """Test the example scheduler executes all the expected surveys"""
        mjd_start = survey_start_mjd()
        scheduler = example_scheduler(mjd_start=mjd_start)
        observatory, scheduler, observations = run_sched(scheduler, mjd_start=mjd_start, survey_length=7)
        u_notes = np.unique(observations["scheduler_note"])

        # Note that some of these may change and need to be updated if
        # survey start date changes, e.g., different DDFs in season,
        # or different lunar phase means different filters get picked
        # for the blobs
        notes_to_check = [
            "DD:COSMOS",
            "blob_long, gr, a",
            "blob_long, gr, b",
            "greedy",
            "long",
            "pair_15, iz, a",
            "pair_15, iz, b",
            "pair_15, ri, a",
            "pair_15, ri, b",
            "pair_15, yy, a",
            "pair_15, yy, b",
            "pair_33, gr, a",
            "pair_33, gr, b",
            "pair_33, ri, a",
            "pair_33, ug, a",
            "pair_33, ug, b",
            "pair_33, yy, a",
            "pair_33, yy, b",
            "pair_33, zy, a",
            "pair_33, zy, b",
            "twilight_near_sun, 0",
            "twilight_near_sun, 1",
            "twilight_near_sun, 2",
            "twilight_near_sun, 3",
        ]

        for note in notes_to_check:
            assert note in u_notes

        for note in u_notes:
            # If this fails, time to add something to notes_to_check
            assert note in u_notes


if __name__ == "__main__":
    unittest.main()
