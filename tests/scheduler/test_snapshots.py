import gzip
import os
import pickle
import unittest
from glob import glob
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.example import example_scheduler, run_sched
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.utils import SURVEY_START_MJD


class TestSnapshots(unittest.TestCase):
    @unittest.skip("Test too slow to run routinely.")
    @unittest.skipUnless(
        os.path.isfile(os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")),
        "Test data not available.",
    )
    def test_snapshots(self):
        """Test the saving of scheduler snapshots"""
        scheduler = example_scheduler(mjd_start=SURVEY_START_MJD, nside=8)

        # Find a start time definitely in the middle of the night
        mjd_start = ceil(SURVEY_START_MJD) + 0.15

        # only run for a few minutes
        survey_length = 4.0 / (60 * 60)

        with TemporaryDirectory() as test_dir:
            run_sched(
                scheduler,
                mjd_start=mjd_start,
                survey_length=survey_length,
                nside=scheduler.nside,
                snapshot_dir=test_dir,
            )
            snapshot_files = glob("sched_snapshot_*.p.gz", root_dir=test_dir)
            assert len(snapshot_files) > 0
            test_snapshot = Path(test_dir).joinpath(snapshot_files[0])
            with gzip.open(test_snapshot, "rb") as pio:
                scheduler_snapshot, conditions_snapshot = pickle.load(pio)
            assert isinstance(scheduler_snapshot, CoreScheduler)
            assert isinstance(conditions_snapshot, Conditions)
