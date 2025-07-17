import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.utils.consdb import ConsDBVisits, load_consdb_visits


class TestConsdb(unittest.TestCase):

    @unittest.skip("avoid requiring access to consdb for tests.")
    def test_consdb_read_visits_lsstcam(self):
        day_obs: str = "2025-07-01"
        consdb_visits: ConsDBVisits = load_consdb_visits("lsstcam", day_obs)
        schema_converter: SchemaConverter = SchemaConverter()

        opsim: pd.DataFrame = consdb_visits.opsim
        schema_converter.opsimdf2obs(opsim)

        obs: np.recarray = consdb_visits.obs
        schema_converter.obs2opsim(obs)

    @unittest.skip("avoid requiring access to consdb for tests.")
    def test_consdb_read_visits_with_constraints(self):
        day_obs: str = "2025-07-01"
        constraints = "band = 'z'"
        consdb_visits: ConsDBVisits = load_consdb_visits("lsstcam", day_obs, constraints=constraints)
        opsim: pd.DataFrame = consdb_visits.opsim
        assert len(opsim) > 0
        assert np.all(opsim.band == "z")

    @unittest.skip("avoid requiring access to consdb for tests.")
    def test_consdb_read_visits_comcamsim(self):
        day_obs: str = "2024-06-26"
        consdb_visits: ConsDBVisits = load_consdb_visits("lsstcomcamsim", day_obs)
        schema_converter: SchemaConverter = SchemaConverter()

        opsim: pd.DataFrame = consdb_visits.opsim
        schema_converter.opsimdf2obs(opsim)

        obs: np.recarray = consdb_visits.obs
        schema_converter.obs2opsim(obs)

    @unittest.skip("avoid requiring access to consdb for tests.")
    def test_consdb_read_visits_comcam(self):
        day_obs: str = "2024-12-10"
        consdb_visits: ConsDBVisits = load_consdb_visits("lsstcomcam", day_obs)
        schema_converter: SchemaConverter = SchemaConverter()

        opsim: pd.DataFrame = consdb_visits.opsim
        schema_converter.opsimdf2obs(opsim)

        obs: np.recarray = consdb_visits.obs
        schema_converter.obs2opsim(obs)

    @unittest.skip("avoid requiring access to consdb for tests.")
    def test_consdb_read_visits_latiss(self):
        day_obs: str = "2025-03-03"
        consdb_visits: ConsDBVisits = load_consdb_visits("latiss", day_obs)
        schema_converter: SchemaConverter = SchemaConverter()

        opsim: pd.DataFrame = consdb_visits.opsim
        schema_converter.opsimdf2obs(opsim)

        obs: np.recarray = consdb_visits.obs
        schema_converter.obs2opsim(obs)

    @unittest.skip("avoid requiring access to consdb for tests.")
    def test_consdb_read_visits_with_token_file(self):
        if "ACCESS_TOKEN" in os.environ:
            token = os.environ["ACCESS_TOKEN"]
        else:
            try:
                import lsst.rsp

                token = lsst.rsp.get_access_token()
            except ImportError:
                return

        with TemporaryDirectory() as temp_dir:
            token_file = Path(temp_dir).joinpath("test_file")
            with open(token_file, "w") as token_io:
                token_io.write(token)

            day_obs: str = "2025-06-20"
            consdb_visits: ConsDBVisits = load_consdb_visits("lsstcam", day_obs, token_file=token_file)
            schema_converter: SchemaConverter = SchemaConverter()

            opsim: pd.DataFrame = consdb_visits.opsim
            schema_converter.opsimdf2obs(opsim)

            obs: np.recarray = consdb_visits.obs
            schema_converter.obs2opsim(obs)
