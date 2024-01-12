import unittest

import healpy as hp
import numpy as np
import pandas as pd

import rubin_scheduler.scheduler.basis_functions as basis_functions
import rubin_scheduler.scheduler.surveys as surveys
from rubin_scheduler.scheduler.basis_functions import SimpleArrayBasisFunction
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.utils import empty_observation, set_default_nside


class TestSurveys(unittest.TestCase):
    def test_field_survey(self):
        nside = 32

        bfs = []
        bfs.append(basis_functions.M5DiffBasisFunction(nside=nside))
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, reward_value=1)

        observatory = ModelObservatory()

        # Check dunder methods
        self.assertIsInstance(repr(survey), str)
        self.assertIsInstance(str(survey), str)

        # Check survey access methods
        conditions = observatory.return_conditions()
        reward = survey.calc_reward_function(conditions)
        self.assertIsInstance(reward, float)
        reward_df = survey.reward_changes(conditions)
        reward_df = survey.make_reward_df(conditions)
        self.assertIsInstance(reward_df, pd.DataFrame)
        reward_df = survey.make_reward_df(conditions, accum=False)

    def test_pointings_survey(self):
        """Test the pointing survey."""
        mo = ModelObservatory()
        conditions = mo.return_conditions()

        # Make a ring of points near the equator so
        # some should always be visible
        fields = empty_observation(n=10)
        fields["RA"] = np.arange(0, fields.size) / fields.size * 2.0 * np.pi
        fields["dec"] = -0.01
        fields["note"] = ["test%i" % ind for ind in range(fields.size)]
        fields["filter"] = "r"
        survey = surveys.PointingsSurvey(fields)

        reward = survey.calc_reward_function(conditions)
        assert np.isfinite(reward)

        obs = survey.generate_observations(conditions)
        # Confirm that our desired input values got passed through
        assert obs[0]["dec"] < 0
        assert obs[0]["note"][0][0:4] == "test"

        # Adding observations
        assert np.sum(survey.n_obs) == 0
        survey.add_observation(obs[0])
        assert np.sum(survey.n_obs) == 1
        survey.add_observations_array(fields, None)
        assert np.sum(survey.n_obs) == 11

        # Check we can get display things out
        rc = survey.reward_changes(conditions)
        assert len(rc) == len(survey.weights)

        # Check we get a dataFrame
        df = survey.make_reward_df(conditions)
        assert len(df) == len(survey.weights)

    def test_roi(self):
        random_seed = 6563
        infeasible_hpix = 123
        nside = set_default_nside()
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(seed=random_seed)
        num_bfs = 3
        bf_values = rng.random((num_bfs, npix))
        bf_values[:, infeasible_hpix] = -np.inf
        bfs = [SimpleArrayBasisFunction(values) for values in bf_values]

        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        # A few cases with an ROI with one valid healpix
        for i in range(3):
            hpix = rng.integers(npix)
            ra, decl = hp.pix2ang(nside, hpix, lonlat=True)
            survey = surveys.FieldSurvey(bfs, RA=ra, dec=decl, reward_value=1)
            reward_df = survey.make_reward_df(conditions)
            for value, max_basis_reward in zip(bf_values[:, hpix], reward_df["max_basis_reward"]):
                self.assertEqual(max_basis_reward, value)

        # One case with an ROI with only an infeasible healpix
        ra, decl = hp.pix2ang(nside, infeasible_hpix, lonlat=True)
        survey = surveys.FieldSurvey(bfs, RA=ra, dec=decl, reward_value=1)
        reward_df = survey.make_reward_df(conditions)
        for max_basis_reward in reward_df["max_basis_reward"]:
            self.assertEqual(max_basis_reward, -np.inf)

        for area in reward_df["basis_area"]:
            self.assertEqual(area, 0.0)

        for feasible in reward_df["feasible"]:
            self.assertFalse(feasible)

        # Make sure it still works as expected if no ROI is set
        weights = [1] * num_bfs
        survey = surveys.BaseMarkovSurvey(bfs, weights)
        reward_df = survey.make_reward_df(conditions)
        for value, max_basis_reward in zip(bf_values.max(axis=1), reward_df["max_basis_reward"]):
            self.assertEqual(max_basis_reward, value)


if __name__ == "__main__":
    unittest.main()
