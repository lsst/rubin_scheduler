import unittest
from copy import deepcopy

import numpy as np

import rubin_scheduler.scheduler.features as features
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.utils import HpInLsstFov, empty_observation
from rubin_scheduler.skybrightness_pre import dark_sky
from rubin_scheduler.utils import survey_start_mjd


class TestFeatures(unittest.TestCase):
    def test_pair_in_night(self):
        pin = features.PairInNight(gap_min=25.0, gap_max=45.0)
        self.assertEqual(np.max(pin.feature), 0.0)

        indx = np.array([1000])

        delta = 30.0 / 60.0 / 24.0

        # Add 1st observation, feature should still be zero
        obs = empty_observation()
        obs["filter"] = "r"
        obs["mjd"] = 59000.0
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 0.0)

        # Add 2nd observation
        obs["mjd"] += delta
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 1.0)

        obs["mjd"] += delta
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 2.0)

    def test_conditions(self):
        observatory = ModelObservatory(init_load_length=1, mjd_start=survey_start_mjd())
        conditions = observatory.return_conditions()
        self.assertIsInstance(repr(conditions), str)
        self.assertIsInstance(str(conditions), str)

        step_days = 1.0

        # Number of sidereal days in a standard day
        sidereal_hours_per_day = 24 * (24.0 / 23.9344696)
        initial_lmst = float(conditions.lmst)
        conditions.mjd = conditions.mjd + step_days
        new_lmst = float(conditions.lmst)
        self.assertAlmostEqual(
            new_lmst,
            (initial_lmst + step_days * sidereal_hours_per_day) % 24,
        )

        # Check that naked conditions work
        conditions_naked = features.Conditions()
        _ = conditions_naked.__str__()

    def test_note_last_observed(self):
        note_last_observed = features.NoteLastObserved(note="test")

        observation = empty_observation()
        observation["mjd"] = 59000.0

        note_last_observed.add_observation(observation=observation)

        assert note_last_observed.feature is None

        observation["note"] = "foo"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature is None

        observation["note"] = "test"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature == observation["mjd"]

    def test_note_last_observed_with_filter(self):
        note_last_observed = features.NoteLastObserved(
            note="test",
            filtername="r",
        )

        observation = empty_observation()
        observation["mjd"] = 59000.0

        note_last_observed.add_observation(observation=observation)

        assert note_last_observed.feature is None

        observation["note"] = "foo"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature is None

        observation["note"] = "test"
        observation["filter"] = "g"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature is None

        observation["note"] = "test"
        observation["filter"] = "r"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature == observation["mjd"]

    def test_NObservationsCurrentSeason(self):
        # Start with basic NObservationsCurrentSeason - no restrictions
        mjd_start = survey_start_mjd()
        nside = 64
        season_feature = features.NObservationsCurrentSeason(nside=nside, mjd_start=mjd_start)
        # Check that season map changes as expected with updates in Conditions
        conditions = features.Conditions(nside=nside)
        conditions.mjd = mjd_start
        season = deepcopy(season_feature.season)
        # with same time, should have same seasons.
        season_feature.season_update(conditions=conditions)
        self.assertTrue(np.all(season_feature.season == season))
        # Advancing time should update seasons in some parts of the sky
        # And leave it the same in other parts of the sky
        conditions.mjd = mjd_start + 100
        season_feature.season_update(conditions=conditions)
        self.assertTrue(np.any(season_feature.season != season))
        self.assertTrue(np.any(season_feature.season == season))
        season = deepcopy(season_feature.season)
        # Do we send a warning if time goes backwards
        # and keep the season and feature the same
        conditions.mjd = mjd_start
        with self.assertWarns(UserWarning):
            season_feature.season_update(conditions=conditions)
        self.assertTrue(np.all(season_feature.season == season))
        # And do the same for observations - check update_seasons works
        # Make some observations
        observations = [empty_observation(), empty_observation()]
        observations[0]["mjd"] = mjd_start
        observations[0]["ID"] = 0
        observations[1]["mjd"] = mjd_start + 100
        observations[1]["ID"] = 1
        season_feature = features.NObservationsCurrentSeason(nside=16, mjd_start=mjd_start)
        season = deepcopy(season_feature.season)
        season_feature.season_update(observation=observations[0])
        self.assertTrue(np.all(season_feature.season == season))
        season_feature.season_update(observation=observations[1])
        self.assertTrue(np.any(season_feature.season != season))
        self.assertTrue(np.any(season_feature.season == season))
        season = deepcopy(season_feature.season)
        # Make time go backwards .. should NOT warn but should not
        # update feature.
        # with self.assertWarns(UserWarning):
        #    season_feature.season_update(observation=observations[0])
        self.assertTrue(np.all(season_feature.season == season))

        # Now check that the _feature) works as expected when we
        # add_observation (i.e. we now care about location and healpix)
        for obs in observations:
            obs["RA"] = np.radians(30)
            obs["dec"] = np.radians(-20)
        observations[1]["mjd"] = mjd_start + 0.3
        pointing2indx = HpInLsstFov(nside=nside)
        indxs = pointing2indx(observations[0]["RA"], observations[0]["dec"])
        # first with no restrictions
        season_feature = features.NObservationsCurrentSeason(nside=nside, mjd_start=mjd_start)
        self.assertTrue(np.all(season_feature.feature == 0))
        season_feature.add_observation(observations[0], indxs)
        # Assert that we added an observation at each of the indexes
        self.assertTrue(np.all(season_feature.feature[indxs] == 1))
        # and nowhere else
        self.assertTrue(np.all(np.delete(season_feature.feature, indxs) == 0))
        # Add another observation within a day
        season_feature.add_observation(observations[1], indxs)
        # Assert that we added an observation at each of the indexes
        self.assertTrue(np.all(season_feature.feature[indxs] == 2))
        # and nowhere else
        self.assertTrue(np.all(np.delete(season_feature.feature, indxs) == 0))
        # Add an observation at a different point on the sky, but where
        # season should not turn oer for ra above yet.
        observations.append(empty_observation())
        observations[-1]["mjd"] = mjd_start + 10
        observations[-1]["RA"] = np.radians(50)
        observations[-1]["dec"] = np.radians(-20)
        observations[-1]["ID"] = len(observations) - 1
        indxs2 = pointing2indx(observations[-1]["RA"], observations[-1]["dec"])
        season_feature.add_observation(observations[-1], indxs2)
        self.assertTrue(np.all(season_feature.feature[indxs2] == 1))
        self.assertTrue(np.all(season_feature.feature[indxs] == 2))
        # Check that if we fast-forward to a future year, and add an
        # observation that everything reset to 1 or 0.
        observation = observations[0]
        observation["mjd"] = mjd_start + 365.25 * 2
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature[indxs] == 1))
        self.assertTrue(np.all(np.delete(season_feature.feature, indxs) == 0))

        # Set up to check add_observations_array - use observations from above.
        observations[0]["mjd"] = mjd_start
        observations_array = np.empty(len(observations), dtype=observations[0].dtype)
        for i, obs in enumerate(observations):
            observations_array[i] = obs
        # Build observations_hpids_array.
        # Find list of lists of healpixels
        # (should match [indxs, indxs, indxs2] from above)
        list_of_hpids = pointing2indx(observations_array["RA"], observations_array["dec"])
        indxs_list = [indxs, indxs, indxs2]
        for hi, ii in zip(list_of_hpids, indxs_list):
            self.assertTrue(set(hi) == set(ii))
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
        # Now check that adding observations to feature via
        # add_observations_array results in the same feature
        feature_obs = features.NObservationsCurrentSeason(nside=nside, mjd_start=mjd_start)
        for obs, idx in zip(observations, indxs_list):
            feature_obs.add_observation(obs, idx)
        feature_arr = features.NObservationsCurrentSeason(nside=nside, mjd_start=mjd_start)
        feature_arr.add_observations_array(observations_array, observations_hpids_array)
        self.assertTrue(np.all(feature_obs.season == feature_arr.season))
        self.assertTrue(np.all(feature_obs.feature == feature_arr.feature))

        # Check that feature works as expected when adding requirements.
        # Add seeing requirement.
        season_feature = features.NObservationsCurrentSeason(
            nside=nside, mjd_start=mjd_start, seeing_fwhm_max=1.0
        )
        observation = empty_observation()
        observation["mjd"] = mjd_start
        observation["RA"] = np.radians(30)
        observation["dec"] = np.radians(-20)
        # Don't add if seeing is too bad
        observation["FWHMeff"] = 1.5
        pointing2indx = HpInLsstFov(nside=nside)
        indxs = pointing2indx(observation["RA"], observation["dec"])
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature == 0))
        # Do add if seeing is good enough
        observation["FWHMeff"] = 0.5
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature[indxs] == 1))
        # Add filter requirement.
        season_feature = features.NObservationsCurrentSeason(
            nside=nside, mjd_start=mjd_start, seeing_fwhm_max=1.0, filtername="r"
        )
        observation["filter"] = "g"
        # Don't add if in the wrong filter
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature == 0))
        # Do add if correct filter and good seeing.
        observation["filter"] = "r"
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature[indxs] == 1))
        # Add m5 requirement.
        season_feature = features.NObservationsCurrentSeason(
            nside=nside, mjd_start=mjd_start, seeing_fwhm_max=1.0, filtername="r", m5_penalty_max=0
        )
        dark_map = dark_sky(nside)["r"]
        # Don't add if too bright
        observation["fivesigmadepth"] = dark_map[indxs].min() - 10
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature == 0))
        # Do add if faint enough
        observation["fivesigmadepth"] = dark_map[indxs].max() + 10
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature[indxs] == 1))
        # Should test also that add_observations_array works
        # in these cases with added requirements .. but will leave it
        # to the "restore" test in test_utils.py.


if __name__ == "__main__":
    unittest.main()
