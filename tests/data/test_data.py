import unittest

from rubin_scheduler.data import data_versions, get_data_dir


class DataTest(unittest.TestCase):
    def testData(self):
        """
        Check that basic data tools work
        """
        data_dir = get_data_dir()
        versions = data_versions()
        assert data_dir is not None
        assert versions is not None


if __name__ == "__main__":
    unittest.main()
