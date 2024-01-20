import os
import unittest

import numpy as np

import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
import rubin_scheduler.utils as utils
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.scheduler.surveys import (
    BlobSurvey,
    GreedySurvey,
    ScriptedSurvey,
    generate_ddf_scheduled_obs,
)
from rubin_scheduler.scheduler.utils import SkyAreaGenerator, calc_norm_factor_array

SAMPLE_BIG_DATA_FILE = os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")


class ModelObservatoryWindy(ModelObservatory):
    """Have the model observatory always have a strong wind from the north"""

    def return_conditions(self):
        """
        Returns
        -------
        rubin_scheduler.scheduler.features.conditions object
        """
        _conditions = super().return_conditions()

        # Always have a strong wind from the north
        wind_speed = 40.0
        wind_direction = 0.0
        self.conditions.wind_speed = wind_speed
        self.conditions.wind_direction = wind_direction

        return self.conditions


def ddf_surveys(detailers=None, season_unobs_frac=0.2, euclid_detailers=None, nside=None):
    obs_array = generate_ddf_scheduled_obs(season_unobs_frac=season_unobs_frac)

    euclid_obs = np.where((obs_array["note"] == "DD:EDFS_b") | (obs_array["note"] == "DD:EDFS_a"))[0]
    all_other = np.where((obs_array["note"] != "DD:EDFS_b") & (obs_array["note"] != "DD:EDFS_a"))[0]

    survey1 = ScriptedSurvey([bf.AvoidDirectWind(nside=nside)], detailers=detailers)
    survey1.set_script(obs_array[all_other])

    survey2 = ScriptedSurvey([bf.AvoidDirectWind(nside=nside)], detailers=euclid_detailers)
    survey2.set_script(obs_array[euclid_obs])

    return [survey1, survey2]


def gen_greedy_surveys(nside):
    """
    Make a quick set of greedy surveys
    """
    sky = SkyAreaGenerator(nside=nside)
    target_map, labels = sky.return_maps()
    filters = ["g", "r", "i", "z", "y"]
    surveys = []

    for filtername in filters:
        bfs = []
        bfs.append(bf.M5DiffBasisFunction(filtername=filtername, nside=nside))
        bfs.append(
            bf.TargetMapBasisFunction(
                filtername=filtername,
                target_map=target_map[filtername],
                out_of_bounds_val=np.nan,
                nside=nside,
            )
        )
        bfs.append(bf.SlewtimeBasisFunction(filtername=filtername, nside=nside))
        bfs.append(bf.StrictFilterBasisFunction(filtername=filtername))
        # Masks, give these 0 weight
        bfs.append(bf.AvoidDirectWind(nside=nside))
        bfs.append(bf.AltAzShadowMaskBasisFunction(nside=nside, shadow_minutes=60.0, max_alt=76.0))
        bfs.append(bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=30.0))
        bfs.append(bf.CloudedOutBasisFunction())

        bfs.append(bf.FilterLoadedBasisFunction(filternames=filtername))
        bfs.append(bf.PlanetMaskBasisFunction(nside=nside))

        weights = np.array([3.0, 0.3, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        surveys.append(
            GreedySurvey(
                bfs,
                weights,
                block_size=1,
                filtername=filtername,
                dither=True,
                nside=nside,
                survey_name="greedy",
            )
        )
    return surveys


def gen_blob_surveys(nside):
    """
    make a quick set of blob surveys
    """
    sky = SkyAreaGenerator(nside=nside)
    target_map, labels = sky.return_maps()
    norm_factor = calc_norm_factor_array(target_map)

    filter1s = ["g"]  # , 'r', 'i', 'z', 'y']
    filter2s = ["g"]  # , 'r', 'i', None, None]

    pair_surveys = []
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        bfs = []
        bfs.append(bf.M5DiffBasisFunction(filtername=filtername, nside=nside))
        if filtername2 is not None:
            bfs.append(bf.M5DiffBasisFunction(filtername=filtername2, nside=nside))
        bfs.append(
            bf.TargetMapBasisFunction(
                filtername=filtername,
                target_map=target_map[filtername],
                out_of_bounds_val=np.nan,
                nside=nside,
                norm_factor=norm_factor,
            )
        )
        if filtername2 is not None:
            bfs.append(
                bf.TargetMapBasisFunction(
                    filtername=filtername2,
                    target_map=target_map[filtername2],
                    out_of_bounds_val=np.nan,
                    nside=nside,
                    norm_factor=norm_factor,
                )
            )
        bfs.append(bf.SlewtimeBasisFunction(filtername=filtername, nside=nside))
        bfs.append(bf.StrictFilterBasisFunction(filtername=filtername))
        # Masks, give these 0 weight
        bfs.append(bf.AvoidDirectWind(nside=nside))
        bfs.append(bf.AltAzShadowMaskBasisFunction(nside=nside, shadow_minutes=60.0, max_alt=76.0))
        bfs.append(bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=30.0))
        bfs.append(bf.CloudedOutBasisFunction())
        # feasibility basis fucntions. Also give zero weight.
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append(bf.FilterLoadedBasisFunction(filternames=filternames))
        bfs.append(bf.TimeToTwilightBasisFunction(time_needed=22.0))
        bfs.append(bf.NotTwilightBasisFunction())
        bfs.append(bf.PlanetMaskBasisFunction(nside=nside))

        weights = np.array([3.0, 3.0, 0.3, 0.3, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if filtername2 is None:
            # Need to scale weights up so filter balancing works properly.
            weights = np.array([6.0, 0.6, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if filtername2 is None:
            survey_name = "blob, %s" % filtername
        else:
            survey_name = "blob, %s%s" % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(filtername=filtername2))

        detailer_list.append(detailers.FlushByDetailer())
        pair_surveys.append(
            BlobSurvey(
                bfs,
                weights,
                filtername1=filtername,
                filtername2=filtername2,
                survey_note=survey_name,
                ignore_obs="DD",
                detailers=detailer_list,
                nside=nside,
            )
        )
    return pair_surveys


class TestExample(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_example(self):
        """Try out the example scheduler."""
        mjd_start = utils.survey_start_mjd()
        nside = 32
        survey_length = 4.0  # days
        scheduler = example_scheduler(nside=nside, mjd_start=mjd_start)
        observatory = ModelObservatory(nside=nside, mjd_start=mjd_start)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )
        # Check that greedy observed some
        assert "greedy" in observations["note"]
        # check some long pairs got observed
        assert np.any(["pair_33" in obs for obs in observations["note"]])
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0
        # Make sure a DDF executed
        assert np.any(["DD" in note for note in observations["note"]])


class TestFeatures(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_greedy(self):
        """
        Set up a greedy survey and run for a few days.
        A crude way to touch lots of code.
        """
        mjd_start = utils.survey_start_mjd()
        nside = 32
        survey_length = 2.0  # days

        surveys = gen_greedy_surveys(nside)
        # Depreating Pairs_survey_scripted
        # surveys.append(Pairs_survey_scripted(None, ignore_obs='DD'))

        # Set up the DD
        dd_surveys = ddf_surveys(nside=nside)
        surveys.extend(dd_surveys)

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatory(nside=nside, mjd_start=mjd_start)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )

        # Check that greedy observed some
        assert "greedy" in observations["note"]
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0

    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_blobs(self):
        """
        Set up a blob selection survey
        """
        mjd_start = utils.survey_start_mjd()
        nside = 32
        survey_length = 2.0  # days

        surveys = []
        # Set up the DD
        dd_surveys = ddf_surveys(nside=nside)
        surveys.append(dd_surveys)

        surveys.append(gen_blob_surveys(nside))
        surveys.append(gen_greedy_surveys(nside))

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatory(nside=nside, mjd_start=mjd_start)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )

        # Make sure some blobs executed
        assert "blob, gg, b" in observations["note"]
        assert "blob, gg, a" in observations["note"]
        # Make sure some greedy executed
        assert "greedy" in observations["note"]
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0

    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_wind(self):
        """
        Test that a wind mask prevent things from being executed in the wrong spot
        """
        mjd_start = utils.survey_start_mjd()
        nside = 32
        survey_length = 4.0  # days

        surveys = []
        # Set up the DD
        dd_surveys = ddf_surveys(nside=nside)
        surveys.append(dd_surveys)

        surveys.append(gen_blob_surveys(nside))
        surveys.append(gen_greedy_surveys(nside))

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatoryWindy(nside=nside, mjd_start=mjd_start)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )

        # Make sure some blobs executed
        assert "blob, gg, b" in observations["note"]
        assert "blob, gg, a" in observations["note"]
        # Make sure some greedy executed
        assert "greedy" in observations["note"]
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0

        # Make sure nothing executed in the strong wind
        assert np.min(np.degrees(observations["az"])) > 30.0
        assert np.max(np.degrees(observations["az"])) < (360.0 - 30.0)

    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_nside(self):
        """
        test running at higher nside
        """
        mjd_start = utils.survey_start_mjd()
        nside = 64
        survey_length = 2.0  # days

        surveys = []
        # Set up the DD
        dd_surveys = ddf_surveys(nside=nside)
        surveys.append(dd_surveys)

        surveys.append(gen_blob_surveys(nside))
        surveys.append(gen_greedy_surveys(nside))

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatory(nside=nside, mjd_start=mjd_start)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )

        # Make sure some blobs executed
        assert "blob, gg, b" in observations["note"]
        assert "blob, gg, a" in observations["note"]
        # Make sure some greedy executed
        assert "greedy" in observations["note"]
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0


if __name__ == "__main__":
    unittest.main()
