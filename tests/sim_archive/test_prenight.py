import unittest
from tempfile import TemporaryDirectory

try:
    from rubin_scheduler.sim_archive import prenight_sim_cli # type: ignore
    from lsst.resources import ResourcePath
    from rubin_sim.data import get_baseline
    HAVE_RESOURCES = True
except ModuleNotFoundError:
    HAVE_RESOURCES = False

class TestPrenight(unittest.TestCase):

    @unittest.skip("Too slow")
    @unittest.skipIf(not HAVE_RESOURCES, "No lsst.resources")
    def test_prenight(self):
        with TemporaryDirectory() as test_archive_dir:
            archive_uri = ResourcePath(test_archive_dir).geturl()
            prenight_sim_cli("--archive", archive_uri)

