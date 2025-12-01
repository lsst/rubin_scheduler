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


if __name__ == "__main__":
    unittest.main()
