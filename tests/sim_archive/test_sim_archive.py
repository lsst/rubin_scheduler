import lzma
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.sim_archive.sim_archive import (
    check_opsim_archive_resource,
    make_sim_archive_dir,
    transfer_archive_dir,
)
from rubin_scheduler.utils import survey_start_mjd


class TestSimArchive(unittest.TestCase):
    def test_sim_archive(self):
        # Begin by running a short simulation
        mjd_start = survey_start_mjd()
        survey_length = 1  # days
        scheduler = example_scheduler(mjd_start=mjd_start)
        scheduler.keep_rewards = True
        observatory = ModelObservatory(mjd_start=mjd_start)

        # Record the state of the scheduler at the start of the sim.
        data_dir = TemporaryDirectory()
        data_path = Path(data_dir.name)

        scheduler_fname = data_path.joinpath("scheduler.pickle.xz")
        with lzma.open(scheduler_fname, "wb", format=lzma.FORMAT_XZ) as pio:
            pickle.dump(scheduler, pio)

        files_to_archive = {"scheduler": scheduler_fname}

        # Run the simulation
        sim_runner_kwargs = {
            "mjd_start": mjd_start,
            "survey_length": survey_length,
            "record_rewards": True,
        }

        observatory, scheduler, observations, reward_df, obs_rewards = sim_runner(
            observatory, scheduler, **sim_runner_kwargs
        )

        # Make the scratch sim archive
        make_sim_archive_dir(
            observations,
            reward_df=reward_df,
            obs_rewards=obs_rewards,
            in_files=files_to_archive,
            sim_runner_kwargs=sim_runner_kwargs,
            data_path=data_path,
        )

        # Move the scratch sim archive to a test resource
        test_resource_dir = TemporaryDirectory()
        test_resource_uri = "file://" + test_resource_dir.name
        sim_archive_uri = transfer_archive_dir(data_dir.name, test_resource_uri)

        # Check the saved archive
        archive_check = check_opsim_archive_resource(sim_archive_uri)
        self.assertEqual(
            archive_check.keys(),
            set(
                [
                    "opsim.db",
                    "rewards.h5",
                    "scheduler.pickle.xz",
                    "obs_stats.txt",
                    "environment.txt",
                    "pypi.json",
                ]
            ),
        )
        for value in archive_check.values():
            self.assertTrue(value)
