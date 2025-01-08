import unittest

import numpy as np
import pandas as pd

from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.utils.consdb import ConsDBVisits, load_consdb_visits


class TestConsdb(unittest.TestCase):
    @unittest.skip("avoid requiring access to consdb for tests.")
    def test_consdb_read_visits_quick(self):
        day_obs: str = "2024-06-26"
        consdb_visits: ConsDBVisits = load_consdb_visits("lsstcomcamsim", day_obs)
        schema_converter: SchemaConverter = SchemaConverter()

        opsim: pd.DataFrame = consdb_visits.opsim
        schema_converter.opsimdf2obs(opsim)

        obs: np.recarray = consdb_visits.obs
        schema_converter.obs2opsim(obs)
