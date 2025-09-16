import unittest
from copy import deepcopy

import numpy as np

import rubin_scheduler.scheduler.features as features
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.utils import HpInLsstFov, ObservationArray
from rubin_scheduler.skybrightness_pre import dark_sky
from rubin_scheduler.utils import SURVEY_START_MJD


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
    return observations_array, observations_hpids_array, list_of_hpids


class TestFeatures(unittest.TestCase):
    def test_features_add_observation_methods(self):
        # Generic test that add_observations_array equals add_observations
        # Under the default conditions.
        # Tests for each feature should still check on non-default kwargs
        observations_list = make_observations_list(20)
        for i, obs in enumerate(observations_list):
            if i % 2 == 0:
                observations_list[i]["dec"] = np.radians(-10)
            if i % 3 == 0:
                observations_list[i]["scheduler_note"] = "a"
            if (i - 1) % 3 == 0:
                observations_list[i]["scheduler_note"] = "b"
            if i > 10:
                observations_list[i]["band"] = "g"
        observations_array, observations_hpid, list_of_hpids = make_observations_arrays(observations_list)
        features_to_test = features.BaseSurveyFeature.__subclasses__()
        for ff in features_to_test:
            test_featureA = ff()
            for obs, indx in zip(observations_list, list_of_hpids):
                test_featureA.add_observation(obs, indx=indx)
            test_featureB = ff()
            test_featureB.add_observations_array(observations_array, observations_hpid)
            print(test_featureA.__class__.__name__)
            if isinstance(test_featureA.feature, (float, int)):
                self.assertTrue(test_featureA.feature == test_featureB.feature)
            elif isinstance(test_featureA.feature, ObservationArray):
                self.assertTrue(test_featureA.feature == test_featureB.feature)
            else:
                # Test nans (if present) are in the same places
                nanA = np.where(np.isnan(test_featureA.feature), 1, 0)
                nanB = np.where(np.isnan(test_featureB.feature), 1, 0)
                self.assertTrue(np.all(nanA == nanB))
                # Test equal where not nan
                self.assertTrue(
                    np.all(
                        test_featureA.feature[np.where(nanA == 0)]
                        == test_featureB.feature[np.where(nanA == 0)]
                    )
                )

    def test_pair_in_night(self):
        pin = features.PairInNight(gap_min=25.0, gap_max=45.0)
        self.assertEqual(np.max(pin.feature), 0.0)

        indx = np.array([1000])

        delta = 30.0 / 60.0 / 24.0

        # Add 1st observation, feature should still be zero
        obs = ObservationArray()
        obs["band"] = "r"
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

    def test_note_in_night(self):
        obs = ObservationArray()

        plain_feature = features.NoteInNight()
        plain_feature.add_observation(obs)

        assert plain_feature.feature == 1

        note_feature = features.NoteInNight(notes=["asdafas", "widget"])
        note_feature.add_observation(obs)

        assert note_feature.feature == 0

        obs["scheduler_note"] = "asdafas"
        note_feature.add_observation(obs)

        assert note_feature.feature == 1

        # should only match if exact
        obs["scheduler_note"] = "asdafa"
        note_feature.add_observation(obs)

        assert note_feature.feature == 1

        obs["scheduler_note"] = "widget"
        note_feature.add_observation(obs)

        assert note_feature.feature == 2

    def test_conditions(self):
        observatory = ModelObservatory(init_load_length=1, mjd_start=SURVEY_START_MJD)
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
        observations_list = make_observations_list(5)
        for i, obs in enumerate(observations_list):
            if i == 1:
                obs["scheduler_note"] = "test 2"
            if i > 3:
                obs["band"] = "g"
        observations_array, observations_hpid, list_of_hpids = make_observations_arrays(observations_list)

        # Test with no note - match all
        note_last_observed = features.LastObservationMjd(scheduler_note=None)
        for obs in observations_list:
            note_last_observed.add_observation(observation=obs)
        self.assertTrue(note_last_observed.feature == observations_list[-1]["mjd"])

        note_last_observed = features.LastObservationMjd(scheduler_note=None)
        note_last_observed.add_observations_array(observations_array, observations_hpid)
        self.assertTrue(note_last_observed.feature == observations_list[-1]["mjd"])

        # Test not matching the note
        note_last_observed = features.LastObservationMjd(scheduler_note="special")
        for obs in observations_list:
            note_last_observed.add_observation(observation=obs)
        self.assertTrue(note_last_observed.feature is None)

        note_last_observed = features.LastObservationMjd(scheduler_note="special")
        note_last_observed.add_observations_array(observations_array, observations_hpid)
        self.assertTrue(note_last_observed.feature is None)

        # Test matching the note
        note_last_observed = features.LastObservationMjd(scheduler_note="test")
        for obs in observations_list:
            note_last_observed.add_observation(observation=obs)
        self.assertTrue(note_last_observed.feature == observations_list[-1]["mjd"])

        note_last_observed = features.LastObservationMjd(scheduler_note="test")
        note_last_observed.add_observations_array(observations_array, observations_hpid)
        self.assertTrue(note_last_observed.feature == observations_list[-1]["mjd"])

        # Add a band requirement
        note_last_observed = features.LastObservationMjd(
            note="test",
            bandname="g",
        )
        for obs in observations_list:
            note_last_observed.add_observation(observation=obs)
        self.assertTrue(note_last_observed.feature == observations_list[-1]["mjd"])

        note_last_observed = features.LastObservationMjd(
            note="test",
            bandname="g",
        )
        note_last_observed.add_observations_array(observations_array, observations_hpid)
        self.assertTrue(note_last_observed.feature == observations_list[-1]["mjd"])

        # Add a different band requirement
        note_last_observed = features.LastObservationMjd(
            note="test",
            bandname="r",
        )
        for obs in observations_list:
            note_last_observed.add_observation(observation=obs)
        self.assertTrue(note_last_observed.feature == observations_list[3]["mjd"])

        note_last_observed = features.LastObservationMjd(
            note="test",
            bandname="r",
        )
        note_last_observed.add_observations_array(observations_array, observations_hpid)
        self.assertTrue(note_last_observed.feature == observations_list[3]["mjd"])

    def test_NObservationsCurrentSeason(self):
        # Start with basic NObservationsCurrentSeason - no restrictions
        mjd_start = SURVEY_START_MJD
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
        observations = [ObservationArray(), ObservationArray()]
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
        observations.append(ObservationArray())
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
        observations_array, observations_hpids_array, list_of_hpids = make_observations_arrays(
            observations, nside=64
        )
        indxs_list = [indxs, indxs, indxs2]
        for hi, ii in zip(list_of_hpids, indxs_list):
            self.assertTrue(set(hi) == set(ii))
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
        observation = ObservationArray()
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
        # Add band requirement.
        season_feature = features.NObservationsCurrentSeason(
            nside=nside, mjd_start=mjd_start, seeing_fwhm_max=1.0, bandname="r"
        )
        observation["band"] = "g"
        # Don't add if in the wrong band
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature == 0))
        # Do add if correct band and good seeing.
        observation["band"] = "r"
        season_feature.add_observation(observation, indxs)
        self.assertTrue(np.all(season_feature.feature[indxs] == 1))
        # Add m5 requirement.
        season_feature = features.NObservationsCurrentSeason(
            nside=nside, mjd_start=mjd_start, seeing_fwhm_max=1.0, bandname="r", m5_penalty_max=0
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

    def test_NObsCount(self):
        # Make some observations to count
        observations_list = make_observations_list(5)
        for i in [0, 2]:
            observations_list[i]["scheduler_note"] = "survey a"
        for i in [1, 3]:
            observations_list[i]["scheduler_note"] = "survey b"
        observations_list[4]["scheduler_note"] = "survey"
        for i in [0, 1]:
            observations_list[i]["band"] = "r"
        for i in [2, 3, 4]:
            observations_list[i]["band"] = "g"
        observations_array, observations_hpid_array, list_of_hpids = make_observations_arrays(
            observations_list
        )
        # Count the observations matching any note
        count_feature = features.NObsCount(scheduler_note=None, bandname=None)
        for obs in observations_list:
            count_feature.add_observation(obs)
        self.assertTrue(count_feature.feature == 5)
        # and count again using add_observations_array
        count_feature = features.NObsCount(scheduler_note=None, bandname=None)
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature == 5)
        # Count using a note to match
        # Count the observations matching specific note
        count_feature = features.NObsCount(scheduler_note="survey a")
        for obs in observations_list:
            count_feature.add_observation(obs)
        self.assertTrue(count_feature.feature == 2)
        # and count again using add_observations_array
        count_feature = features.NObsCount(scheduler_note="survey a")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature == 2)
        # Count the observations matching subset of note
        count_feature = features.NObsCount(scheduler_note="survey")
        for obs in observations_list:
            count_feature.add_observation(obs)
        self.assertTrue(count_feature.feature == 5)
        # and count again using add_observations_array
        count_feature = features.NObsCount(scheduler_note="survey")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature == 5)
        # Count the observations matching band
        count_feature = features.NObsCount(scheduler_note=None, bandname="r")
        for obs in observations_list:
            count_feature.add_observation(obs)
        self.assertTrue(count_feature.feature == 2)
        # and count again using add_observations_array
        count_feature = features.NObsCount(scheduler_note=None, bandname="r")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature == 2)
        # Count the observations matching band and surveyname
        count_feature = features.NObsCount(scheduler_note="survey b", bandname="r")
        for obs in observations_list:
            count_feature.add_observation(obs)
        self.assertTrue(count_feature.feature == 1)
        # and count again using add_observations_array
        count_feature = features.NObsCount(scheduler_note="survey b", bandname="r")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature == 1)

        # Test backward compatibilty shim
        count_feature = features.NObsCount(note="survey b", bandname="r")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature == 1)

    def test_LastObservation(self):
        # Make some observations to count
        observations_list = make_observations_list(4)
        observations_list[0]["mjd"] = SURVEY_START_MJD
        observations_list[1]["mjd"] = SURVEY_START_MJD + 10
        observations_list[2]["mjd"] = SURVEY_START_MJD + 20
        observations_list[3]["mjd"] = SURVEY_START_MJD + 30
        observations_list[0]["scheduler_note"] = "survey a"
        observations_list[1]["scheduler_note"] = "survey"
        observations_list[2]["scheduler_note"] = "survey a"
        observations_list[3]["scheduler_note"] = "survey b"
        observations_array, observations_hpid_array, list_of_hpids = make_observations_arrays(
            observations_list
        )
        # Observations matching any note
        count_feature = features.LastObservation(scheduler_note=None)
        for obs, indx in zip(observations_list, list_of_hpids):
            count_feature.add_observation(obs, indx=indx)
        self.assertTrue(count_feature.feature["mjd"] == observations_list[-1]["mjd"])
        # and count again using add_observations_array
        count_feature = features.LastObservation(scheduler_note=None)
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature["mjd"] == observations_list[-1]["mjd"])
        # Observations matching a specific note.
        count_feature = features.LastObservation(scheduler_note="survey b")
        for obs in observations_list:
            count_feature.add_observation(obs)
        self.assertTrue(count_feature.feature["mjd"] == observations_list[-1]["mjd"])
        # and count again (with different note) using add_observations_array
        count_feature = features.LastObservation(scheduler_note="survey a")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature["mjd"] == observations_list[-2]["mjd"])

        # Observations matching a subset of note.
        count_feature = features.LastObservation(scheduler_note="survey")
        for obs in observations_list:
            count_feature.add_observation(obs)
        self.assertTrue(count_feature.feature["mjd"] == observations_list[-1]["mjd"])
        # and count again using add_observations_array
        count_feature = features.LastObservation(scheduler_note="survey")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature["mjd"] == observations_list[-1]["mjd"])

    def test_last_obs_matching(self):
        """Test the LastObservedMatching"""

        feature = features.LastObservedMatching(scheduler_note="test")

        # Add an observation that should count
        obs = ObservationArray(n=1)
        obs["scheduler_note"] = "test"
        obs["mjd"] = 10
        feature.add_observation(obs)

        assert feature.feature["mjd"] == 10

        # Add observation that should not count
        obs["scheduler_note"] = "test1"
        obs["mjd"] = 20
        feature.add_observation(obs)

        assert feature.feature["mjd"] == 10

        # test add array
        feature = features.LastObservedMatching(scheduler_note="test")
        obs = ObservationArray(n=2)
        obs["scheduler_note"] = "test"
        obs["mjd"] = 20

        feature.add_observations_array(obs, [])
        assert feature.feature == obs[-1]

        # Test distance cut
        feature = features.LastObservedMatching(scheduler_note="test", ra=0, dec=0, ang_distance_match=4.0)
        obs = ObservationArray(n=1)
        obs["scheduler_note"] = "test"
        obs["RA"] = 0
        obs["dec"] = 0
        obs["mjd"] = 30
        feature.add_observation(obs)

        assert feature.feature["mjd"] == 30

        # Add observation beyond distance cut
        obs = ObservationArray(n=1)
        obs["scheduler_note"] = "test"
        obs["RA"] = 0
        obs["dec"] = np.radians(-20)
        obs["mjd"] = 40
        feature.add_observation(obs)

        assert feature.feature["mjd"] == 30

    def test_NObservations(self):
        # Make some observations to count
        observations_list = make_observations_list(12)
        indexes = []
        nside = 32
        pointing2hpindx = HpInLsstFov(nside=nside)
        for i, obs in enumerate(observations_list):
            obs["mjd"] = SURVEY_START_MJD + i
            obs["rotSkyPos"] = 0
            if i < 6:
                obs["band"] = "r"
            else:
                obs["band"] = "g"
            if i % 2 == 0:
                obs["RA"] = np.radians(30)
            else:
                obs["RA"] = np.radians(10)
            if i % 3 == 0:
                obs["scheduler_note"] = "survey a"
            elif i % 3 == 1:
                obs["scheduler_note"] = "survey b"
            else:
                obs["scheduler_note"] = "survey"
            indexes.append(pointing2hpindx(obs["RA"], obs["dec"], rotSkyPos=obs["rotSkyPos"]))
        observations_array, observations_hpid_array, list_of_hpids = make_observations_arrays(
            observations_list
        )
        # Observations matching any note or band
        count_feature = features.NObservations(bandname=None, scheduler_note=None)
        for obs, indx in zip(observations_list, indexes):
            count_feature.add_observation(obs, indx)
        self.assertTrue(count_feature.feature.max() == 6)
        # and count again using add_observations_array
        count_feature = features.NObservations(bandname=None, scheduler_note=None)
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature.max() == 6)
        # Observations matching a specific note - or are partial matches.
        count_feature = features.NObservations(scheduler_note="survey a")
        for obs, indx in zip(observations_list, indexes):
            count_feature.add_observation(obs, indx)
        self.assertTrue(count_feature.feature.max() == 4)
        # and count again using add_observations_array
        # Observations matching a specific note.
        count_feature = features.NObservations(scheduler_note="survey a")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature.max() == 4)
        # Observations matching a subset of note.
        # It's not obvious that this is what this SHOULD do, and it's not
        # used with "note" in the example /baseline scheduler.
        count_feature = features.NObservations(scheduler_note="survey")
        for obs, indx in zip(observations_list, indexes):
            count_feature.add_observation(obs, indx)
        self.assertTrue(count_feature.feature.max() == 2)
        # and count again using add_observations_array
        count_feature = features.NObservations(scheduler_note="survey")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature.max() == 2)
        # Observations matching any note but specified band.
        count_feature = features.NObservations(bandname="r", scheduler_note=None)
        for obs, indx in zip(observations_list, indexes):
            count_feature.add_observation(obs, indx)
        self.assertTrue(count_feature.feature.max() == 3)
        # and count again using add_observations_array
        count_feature = features.NObservations(bandname="r", scheduler_note=None)
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature.max() == 3)
        # Observations matching specific note and  specified band.
        count_feature = features.NObservations(bandname="r", scheduler_note="survey")
        for obs, indx in zip(observations_list, indexes):
            count_feature.add_observation(obs, indx)
        self.assertTrue(count_feature.feature.max() == 1)
        # and count again using add_observations_array
        count_feature = features.NObservations(bandname="r", scheduler_note="survey")
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)
        self.assertTrue(count_feature.feature.max() == 1)

        # Observations matching specific note and specified band and seeing
        fwhms = [0.1] * 6 + [10.0] * 6
        bands = ["r", "g"] * 6
        for obs, fwhm, band in zip(observations_list, fwhms, bands):
            obs["band"] = band
            obs["FWHMeff"] = fwhm
            obs["scheduler_note"] = "survey"

        observations_array, observations_hpid_array, list_of_hpids = make_observations_arrays(
            observations_list
        )

        count_feature = features.NObservations(bandname="r", scheduler_note="survey", seeing_limit=1.0)
        for obs, indx in zip(observations_list, indexes):
            count_feature.add_observation(obs, indx)
        self.assertTrue(count_feature.feature.max() == 3)
        # and count again using add_observations_array
        count_feature = features.NObservations(bandname="r", scheduler_note="survey", seeing_limit=1.0)
        count_feature.add_observations_array(observations_array, observations_hpid=observations_hpid_array)

        self.assertTrue(count_feature.feature.max() == 3)


if __name__ == "__main__":
    unittest.main()
