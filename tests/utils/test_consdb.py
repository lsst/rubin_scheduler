import unittest

import pandas as pd

from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.utils.consdb import ConsDBVisits, load_consdb_visits


class TestConsdb(unittest.TestCase):

    def test_consdb_read_visits_quick(self):
        day_obs: str = "2024-06-26"
        consdb_visits: ConsDBVisits = load_consdb_visits("lsstcomcamsim", day_obs)
        visits: pd.DataFrame = consdb_visits.opsim

        # SchemaConverter has a list of all columns expected to be in obsdb
        expected_columns = set(SchemaConverter().convert_dict)
        returned_columns = set(visits.columns)
        assert expected_columns == returned_columns
