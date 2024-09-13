import unittest

import healpy as hp
import numpy as np
import pandas as pd

import rubin_scheduler.scheduler.basis_functions as basis_functions
import rubin_scheduler.scheduler.surveys as surveys
from rubin_scheduler.scheduler.basis_functions import SimpleArrayBasisFunction
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.utils import HpInLsstFov, ObservationArray, set_default_nside
from rubin_scheduler.utils import survey_start_mjd


def make_observations_list(nobs=1):
    observations_list = []
    for i in range(0, nobs):
        observation = ObservationArray()
        observation["mjd"] = survey_start_mjd() + i * 30 / 60 / 60 / 24
        observation["RA"] = np.radians(30)
        observation["dec"] = np.radians(-20)
        observation["filter"] = "r"
        observation["scheduler_note"] = "test"
        observations_list.append(observation)
    return observations_list


def make_observations_arrays(observations_list, nside=32):
    # Turn list of observations (that should already have useful info)
    # into observations_array plus observations_hpids_array.
    observations_array = np.empty(len(observations_list), dtype=observations_list[0].dtype)
    for i, obs in enumerate(observations_list):
        observations_array[i] = obs
    # Build observations_hpids_array.
    # Find list of lists of healpixels
    # (should match [indxs, indxs, indxs2] from above)
    pointing2indx = HpInLsstFov(nside=nside)
    list_of_hpids = pointing2indx(observations_array["RA"], observations_array["dec"])
    # Unravel list-of-lists (list_of_hpids) to match against observations
    hpids = []
    big_array_indx = []
    for i, indxs in enumerate(list_of_hpids):
        for indx in indxs:
            hpids.append(indx)
            big_array_indx.append(i)
    hpids = np.array(hpids, dtype=[("hpid", int)])
    # Set up format / dtype for observations_hpids_array
    names = list(observations_array.dtype.names)
    types = [observations_array[name].dtype for name in names]
    names.append(hpids.dtype.names[0])
    types.append(hpids["hpid"].dtype)
    ndt = list(zip(names, types))
    observations_hpids_array = np.empty(hpids.size, dtype=ndt)
    # Populate observations_hpid_array - big_array_indx points
    # between index in observations_array and index in hpid
    observations_hpids_array[list(observations_array.dtype.names)] = observations_array[big_array_indx]
    observations_hpids_array[hpids.dtype.names[0]] = hpids
    return observations_array, observations_hpids_array


class TestSurveys(unittest.TestCase):
    def test_field_survey(self):
        nside = 32

        bfs = []
        bfs.append(basis_functions.M5DiffBasisFunction(nside=nside))

        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0)

        observatory = ModelObservatory()

        # Check dunder methods
        self.assertIsInstance(repr(survey), str)
        self.assertIsInstance(str(survey), str)

        # Check survey access methods
        conditions = observatory.return_conditions()
        reward = survey.calc_reward_function(conditions)
        self.assertIsInstance(reward, float)
        reward_df = survey.make_reward_df(conditions)
        self.assertIsInstance(reward_df, pd.DataFrame)
        reward_df = survey.make_reward_df(conditions, accum=False)

    def test_field_survey_add_observations(self):
        nside = 32
        # Just need a placeholder for this.
        bfs = [basis_functions.M5DiffBasisFunction(nside=nside)]

        observations_list = make_observations_list(10)
        indexes = []
        pointing2hpindx = HpInLsstFov(nside=nside)
        for i, obs in enumerate(observations_list):
            obs["mjd"] = survey_start_mjd() + i
            obs["rotSkyPos"] = 0
            if i < 5:
                obs["filter"] = "r"
                obs["scheduler_note"] = "r band"
            else:
                obs["filter"] = "g"
                obs["scheduler_note"] = "g band"
            obs["RA"] = np.radians(90)
            obs["dec"] = np.radians(-30)
            indexes.append(pointing2hpindx(obs["RA"], obs["dec"], rotSkyPos=obs["rotSkyPos"]))
        observations_array, observations_hpid_array = make_observations_arrays(observations_list)

        # Try adding observations to survey one at a time.
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, accept_obs=None)
        for obs, indx in zip(observations_list, indexes):
            survey.add_observation(obs, indx=indx)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == len(observations_list))
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[-1]["mjd"])
        # Try adding observations to survey in array
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, accept_obs=None)
        survey.add_observations_array(observations_array, observations_hpid_array)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == len(observations_list))
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[-1]["mjd"])

        # Now with specific requirements on obs to accept.
        # Try adding observations to survey one at a time.
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, accept_obs=["r band"])
        for obs, indx in zip(observations_list, indexes):
            survey.add_observation(obs, indx=indx)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == 5)
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[4]["mjd"])
        # Try adding observations to survey in array
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, accept_obs=["r band"])
        survey.add_observations_array(observations_array, observations_hpid_array)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == 5)
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[4]["mjd"])
        # Try adding observations to survey one at a time.
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, accept_obs=["r band", "g band"])
        for obs, indx in zip(observations_list, indexes):
            survey.add_observation(obs, indx=indx)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == 10)
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[-1]["mjd"])
        # Try adding observations to survey in array
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, accept_obs=["r band", "g band"])
        survey.add_observations_array(observations_array, observations_hpid_array)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == 10)
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[-1]["mjd"])

    def test_pointings_survey(self):
        """Test the pointing survey."""
        mo = ModelObservatory()
        conditions = mo.return_conditions()

        # Make a ring of points near the equator so
        # some should always be visible
        fields = ObservationArray(n=10)
        fields["RA"] = np.arange(0, fields.size) / fields.size * 2.0 * np.pi
        fields["dec"] = -0.01
        fields["scheduler_note"] = ["test%i" % ind for ind in range(fields.size)]
        fields["filter"] = "r"
        survey = surveys.PointingsSurvey(fields)

        reward = survey.calc_reward_function(conditions)
        assert np.isfinite(reward)

        obs = survey.generate_observations(conditions)
        # Confirm that our desired input values got passed through
        assert obs[0]["dec"] < 0
        assert obs[0]["scheduler_note"][0][0:4] == "test"

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
            survey = surveys.FieldSurvey(bfs, RA=ra, dec=decl)
            reward_df = survey.make_reward_df(conditions)
            for value, max_basis_reward in zip(bf_values[:, hpix], reward_df["max_basis_reward"]):
                self.assertEqual(max_basis_reward, value)

        # One case with an ROI with only an infeasible healpix
        ra, decl = hp.pix2ang(nside, infeasible_hpix, lonlat=True)
        survey = surveys.FieldSurvey(bfs, RA=ra, dec=decl)
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
