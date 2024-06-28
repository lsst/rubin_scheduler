import unittest

from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler
from rubin_scheduler.sim_archive import get_scheduler_instance_from_repo


class TestMakeSnapshot(unittest.TestCase):
    def test_get_scheduler_instance_photcal(self):
        scheduler = get_scheduler_instance_from_repo(
            config_repo="https://github.com/lsst-ts/ts_config_ocs.git",
            config_script="Scheduler/feature_scheduler/auxtel/fbs_config_image_photocal_survey.py",
            config_branch="main",
        )
        self.assertIsInstance(scheduler, CoreScheduler)
