import unittest

import healpy as hp
import numpy as np

from rubin_scheduler.scheduler.example.too_surveys import gen_too_surveys
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.surveys import ToOScriptedSurvey
from rubin_scheduler.scheduler.utils import TargetoO
from rubin_scheduler.utils import hpid2_ra_dec


class TestToO(unittest.TestCase):

    def test_tesselation(self):
        """Test tesselation"""
        survey = ToOScriptedSurvey([])
        ra1, dec1 = survey._tesselate([100, 101])
        # This should spin things
        ra2, dec2 = survey._tesselate([100, 101])

        assert ra2[0] != ra1[0]

        # If we ask for a single HEALpix
        # Should only give a pointing at
        # one location

        ra1, dec1 = survey._tesselate([100])

        assert len(ra1) == 1

        # Should be no change if we call again
        ra2, dec2 = survey._tesselate([100])

        assert ra1 == ra2

    def test_positions(self):
        """Test tesselation rotates as expected"""

        nside = 32
        times = [1, 24, 48]
        bands_at_times = ["i", "i", "i"]
        nvis = [1, 1, 1]
        exptimes = [30, 30, 30]

        too_footprint = np.ones(hp.nside2npix(nside))

        survey = ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=[],
            too_types_to_follow=["test_too"],
            survey_name="ToO, test",
            flushtime=48.0,
            n_snaps=1,
            target_name_base="ToO_test",
            observation_reason="too_test",
        )

        # Generate a ToO event
        target_map = np.zeros(hp.nside2npix(nside))
        ra, dec = hpid2_ra_dec(nside, np.arange(target_map.size))
        target_map[np.where(dec < -85)[0]] = 1
        too_event = TargetoO(
            100, target_map, 6500, 100, ra_rad_center=None, dec_rad_center=None, too_type="test_too"
        )
        conditions = Conditions()
        conditions.targets_of_opportunity = [too_event]

        # Give ToO to survey
        survey.update_conditions(conditions)

        # All pointings should be unique
        n_unique = np.size(np.unique(survey.obs_wanted["RA"]))
        n_all = np.size(survey.obs_wanted)

        assert n_unique > 1
        assert n_unique == n_all

        # Check if we have multiple filters, they are at same position
        bands_at_times = ["ig", "ig", "ig"]
        survey = ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=[],
            too_types_to_follow=["test_too"],
            survey_name="ToO, test",
            flushtime=48.0,
            n_snaps=1,
            target_name_base="ToO_test",
            observation_reason="too_test",
        )

        survey.update_conditions(conditions)

        n_unique = np.size(np.unique(survey.obs_wanted["RA"]))
        n_all = np.size(survey.obs_wanted)

        assert n_unique > 1
        assert n_unique == n_all / 2

        # Check if we have multiple filters, and multiple nvis
        bands_at_times = ["ig", "ig", "ig"]
        nvis = [2, 2, 2]
        survey = ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=[],
            too_types_to_follow=["test_too"],
            survey_name="ToO, test",
            flushtime=48.0,
            n_snaps=1,
            target_name_base="ToO_test",
            observation_reason="too_test",
        )

        survey.update_conditions(conditions)

        n_unique = np.size(np.unique(survey.obs_wanted["RA"]))
        n_all = np.size(survey.obs_wanted)

        assert n_unique > 1
        assert n_unique == n_all / 2

        # Turn dither off
        survey = ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=[],
            too_types_to_follow=["test_too"],
            survey_name="ToO, test",
            flushtime=48.0,
            n_snaps=1,
            target_name_base="ToO_test",
            observation_reason="too_test",
            dither_per_visit=False,
        )

        # Give ToO to survey
        survey.update_conditions(conditions)

        # Now pointings should repeat
        n_unique = np.size(np.unique(survey.obs_wanted["RA"]))
        n_all = np.size(survey.obs_wanted)

        assert n_unique > 1
        assert n_unique < n_all

    def test_events(self):
        """Test the ToO events, make sure each one generates"""

        nside = 32
        too_footprint = np.ones(hp.nside2npix(nside))
        surveys = gen_too_surveys(nside=nside, too_footprint=too_footprint)

        conditions = Conditions()

        target_map = np.zeros(hp.nside2npix(nside))

        ra, dec = hpid2_ra_dec(nside, np.arange(target_map.size))

        target_map[np.where(dec < -85)[0]] = 1

        target_map_single = target_map * 0
        # for lensed_BNS_case_B, it is doing a bad loop, so need
        # to keep footprint very small.
        target_map_single[np.where(target_map == 1)[0].min()] = 1

        for survey in surveys:
            for type in survey.too_types_to_follow:
                tm = target_map
                if type == "lensed_BNS_case_B":
                    tm = target_map_single
                too = TargetoO(100, tm, 6500, 100, ra_rad_center=None, dec_rad_center=None, too_type=type)
                conditions.targets_of_opportunity = [too]

                survey.update_conditions(conditions)
                assert np.size(survey.obs_wanted.shape) > 0
                survey.obs_wanted = np.array([])

    def test_sort(self):
        """Test the sorting works ok"""

        nside = 32
        times = [1, 24, 48]
        bands_at_times = ["i", "i", "i"]
        nvis = [1, 1, 1]
        exptimes = [30, 30, 30]

        too_footprint = np.ones(hp.nside2npix(nside))

        # Check if we have multiple filters, and multiple nvis
        bands_at_times = ["ig", "ig", "ig"]
        nvis = [2, 2, 2]
        survey = ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=[],
            too_types_to_follow=["test_too"],
            survey_name="ToO, test",
            flushtime=48.0,
            n_snaps=1,
            target_name_base="ToO_test",
            observation_reason="too_test",
            return_n_limit=2,
        )

        # Generate a ToO event
        target_map = np.zeros(hp.nside2npix(nside))
        ra, dec = hpid2_ra_dec(nside, np.arange(target_map.size))
        target_map[np.where(dec < -85)[0]] = 1
        too_event = TargetoO(
            100, target_map, 6500, 100, ra_rad_center=None, dec_rad_center=None, too_type="test_too"
        )
        conditions = Conditions()
        conditions.targets_of_opportunity = [too_event]

        # Give ToO to survey
        survey.update_conditions(conditions)
        conditions.mjd = survey.mjd_start + 0.01
        # Sun and moon out of the way
        conditions.moon_ra = 0.0
        conditions.moon_dec = np.pi / 2
        conditions.sun_alt = -np.pi / 2.0

        conditions.tel_alt_limits = [-np.pi / 2, np.pi / 2]
        conditions.tel_az_limits = [-3 * np.pi, 3 * np.pi]
        conditions.mounted_bands = ["u", "g", "r", "i", "z", "y"]

        survey.obs_wanted["HA_min"] = 12
        survey.obs_wanted["HA_max"] = 12

        obs = survey._check_list(conditions)

        assert obs.size == 2

        assert np.size(np.unique(obs["band"])) == 1


if __name__ == "__main__":
    unittest.main()
