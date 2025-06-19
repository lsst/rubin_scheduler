import unittest
import warnings

import healpy as hp
import numpy as np
import pandas as pd

import rubin_scheduler.scheduler.basis_functions as basis_functions
import rubin_scheduler.scheduler.detailers as detailers
import rubin_scheduler.scheduler.surveys as surveys
from rubin_scheduler.scheduler.basis_functions import SimpleArrayBasisFunction
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.scheduler.utils import HpInLsstFov, ObservationArray, ScheduledObservationArray
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD


def make_observations_list(nobs=1):
    observations_list = []
    for i in range(0, nobs):
        observation = ObservationArray()
        observation["mjd"] = SURVEY_START_MJD + i * 30 / 60 / 60 / 24
        observation["RA"] = np.radians(30)
        observation["dec"] = np.radians(-20)
        observation["band"] = "r"
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

    def test_base_survey(self):
        nside = 32

        bfs = []
        bfs.append(basis_functions.MoonAvoidanceBasisFunction(nside=nside))

        detailer_list = []
        science_program = "BLOCK-0"
        survey = surveys.BaseSurvey(
            bfs, detailers=detailer_list, survey_name="test", science_program=science_program
        )

        # Check that the TrackingInfoDetailer was added.
        idx = -1
        for i, det in enumerate(survey.detailers):
            if isinstance(det, detailers.TrackingInfoDetailer):
                idx = i
        self.assertTrue(i > -1)
        self.assertEqual(science_program, survey.detailers[idx].science_program)
        # And there is a warning if it's already present.
        detailer_list = [detailers.TrackingInfoDetailer(science_program="BLOCK-T")]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            surveys.BaseSurvey(
                bfs, detailers=detailer_list, survey_name="test", science_program=science_program
            )
            assert len(w) >= 1

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
        _ = survey.make_reward_df(conditions, accum=False)

    def test_field_survey_add_observations(self):
        nside = 32
        # Just need a placeholder for this.
        bfs = [basis_functions.M5DiffBasisFunction(nside=nside)]

        observations_list = make_observations_list(10)
        indexes = []
        pointing2hpindx = HpInLsstFov(nside=nside)
        for i, obs in enumerate(observations_list):
            obs["mjd"] = SURVEY_START_MJD + i
            obs["rotSkyPos"] = 0
            if i < 5:
                obs["band"] = "r"
                obs["scheduler_note"] = "r band"
            else:
                obs["band"] = "g"
                obs["scheduler_note"] = "g band"
            obs["RA"] = np.radians(90)
            obs["dec"] = np.radians(-30)
            indexes.append(pointing2hpindx(obs["RA"], obs["dec"], rotSkyPos=obs["rotSkyPos"]))
        observations_array, observations_hpid_array = make_observations_arrays(observations_list)

        # Try adding observations to survey one at a time.
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0)
        for obs, indx in zip(observations_list, indexes):
            survey.add_observation(obs, indx=indx)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == len(observations_list))
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[-1]["mjd"])
        # Try adding observations to survey in array
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0)
        survey.add_observations_array(observations_array, observations_hpid_array)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == len(observations_list))
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[-1]["mjd"])

        # Now with different scheduler_notes.
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, scheduler_note="r band")
        for obs, indx in zip(observations_list, indexes):
            survey.add_observation(obs, indx=indx)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == len(observations_list))
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[-1]["mjd"])
        self.assertTrue(survey.extra_features["ObsRecorded_note"].feature == 5)
        self.assertTrue(survey.extra_features["LastObs_note"].feature["mjd"] == observations_list[4]["mjd"])
        # Try adding observations to survey in array
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, scheduler_note="r band")
        survey.add_observations_array(observations_array, observations_hpid_array)
        self.assertTrue(survey.extra_features["ObsRecorded"].feature == len(observations_list))
        self.assertTrue(survey.extra_features["LastObs"].feature["mjd"] == observations_list[-1]["mjd"])
        self.assertTrue(survey.extra_features["ObsRecorded_note"].feature == 5)
        self.assertTrue(survey.extra_features["LastObs_note"].feature["mjd"] == observations_list[4]["mjd"])

    def test_field_altaz(self):
        """Test that we can use alt,az pointings"""
        survey = surveys.FieldAltAzSurvey([], alt=75, az=180.0)
        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        reward = survey.calc_reward_function(conditions)
        self.assertIsInstance(reward, float)
        reward_df = survey.make_reward_df(conditions)
        self.assertIsInstance(reward_df, pd.DataFrame)
        _ = survey.make_reward_df(conditions, accum=False)

        observation = survey.generate_observations(conditions)
        assert observation[0].dtype == ObservationArray().dtype

        # Advance time slightly
        observatory.mjd = observatory.mjd + 0.1
        conditions = observatory.return_conditions()

        observation_2 = survey.generate_observations(conditions)

        # Dec should stay the same, but RA will shift
        assert observation[0]["dec"] == observation_2[0]["dec"]
        assert observation[0]["RA"] != observation_2[0]["RA"]

        # Test that masks are respected
        # Point at a masked altitude. Reward should be nan or -inf
        survey = surveys.FieldAltAzSurvey(
            [basis_functions.AltAzShadowMaskBasisFunction(max_alt=85.0)], alt=90, az=180.0
        )
        reward = survey.calc_reward_function(conditions)
        assert ~np.isfinite(reward)

        # Point at an allowed region. Reward should be finite now.
        survey = surveys.FieldAltAzSurvey(
            [basis_functions.AltAzShadowMaskBasisFunction(max_alt=85.0)], alt=80, az=180.0
        )
        reward = survey.calc_reward_function(conditions)
        assert np.isfinite(reward)

    def test_scripted_survey(self):
        """Test the add observations methods of scripted surveys"""

        survey = surveys.ScriptedSurvey([])

        observations = ScheduledObservationArray(n=5)
        observations["band"] = "r"
        observations["scheduler_note"] = ["a", "a", "c", "d", "a"]
        observations["mjd_tol"] = 1
        observations["mjd"] = 1
        observations["flush_by_mjd"] = 2

        survey.set_script(observations)

        # Make a completed list that has the first 3 observations
        completed_observations = ObservationArray(n=3)
        completed_observations["band"] = survey.obs_wanted["band"][0:3]
        completed_observations["scheduler_note"] = survey.obs_wanted["scheduler_note"][0:3]

        # Add one at a time
        for obs in completed_observations:
            not_a_slice = ObservationArray(n=1)
            for key in obs.dtype.names:
                not_a_slice[key] = obs[key]
            survey.add_observation(not_a_slice)

        assert np.sum(survey.obs_wanted["observed"]) == 3

        # Init to clear the internal counter
        survey = surveys.ScriptedSurvey([])
        observations["scheduler_note"] = ["a", "a", "c", "d", "a"]
        survey.set_script(observations)
        sched = CoreScheduler([survey])
        # Add full array at once
        sched.add_observations_array(completed_observations)
        assert np.sum(survey.obs_wanted["observed"]) == 3

        # Make sure error gets raised if we set a script wrong
        survey = surveys.ScriptedSurvey([])
        observations["scheduler_note"] = ["a", "a", "c", "d", "a"]
        with self.assertRaises(Exception) as context:
            survey.set_script(observations, add_index=False)
            self.assertTrue("unique scheduler_note" in str(context))

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
        fields["band"] = "r"
        survey = surveys.PointingsSurvey(fields)

        reward = survey.calc_reward_function(conditions)
        assert np.isfinite(reward)

        obs = survey.generate_observations(conditions)
        # Confirm that our desired input values got passed through
        assert obs["dec"][0] < 0
        assert obs["scheduler_note"][0][0:4] == "test"

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
        nside = DEFAULT_NSIDE
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
