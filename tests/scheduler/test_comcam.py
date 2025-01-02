import unittest

import healpy as hp
import numpy as np

import rubin_scheduler.scheduler.basis_functions as bf
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.scheduler.surveys import BlobSurvey
from rubin_scheduler.scheduler.utils import Footprint


class TestComCam(unittest.TestCase):
    def test_comcam(self):
        """Make a simple r and g survey to run with ComCam."""
        # Specify the HEALpix resolution we want to do everything in.
        # Need to turn up for comcam
        nside = 512
        mo = ModelObservatory(nside=nside, downtimes="ideal", cloud_data="ideal")
        conditions = mo.return_conditions()

        # Let's start out with a simple 2-band footprint where
        # we want 1 observations in r for every 2 observations in g
        blank_map = np.zeros(hp.nside2npix(nside)) + np.nan
        # cut down to a very narrow band
        indx = np.where((conditions.dec < np.radians(-15)) & (conditions.dec > np.radians(-20)))
        simple_fp = {"r": blank_map + 0, "g": blank_map + 0}
        simple_fp["r"][indx] = 1
        simple_fp["g"][indx] = 1

        fp = Footprint(mo.mjd_start, mo.sun_ra_start, nside=nside)
        for bandname in simple_fp:
            fp.set_footprint(bandname, simple_fp[bandname])

        footprint_weight = 1.0
        m5_weight = 0.0

        detailers = []

        red_fp_basis = bf.FootprintBasisFunction(bandname="r", footprint=fp, nside=nside)
        m5_basis_r = bf.M5DiffBasisFunction(bandname="r", nside=nside)
        red_survey = BlobSurvey(
            [red_fp_basis, m5_basis_r],
            [footprint_weight, m5_weight],
            bandname1="r",
            survey_name="r_blob",
            nside=nside,
            camera="comcam",
            grow_blob=False,
            detailers=detailers,
            dither=False,
            twilight_scale=False,
        )

        blue_fp_basis = bf.FootprintBasisFunction(bandname="g", footprint=fp, nside=nside)
        m5_basis_g = bf.M5DiffBasisFunction(bandname="g", nside=nside)
        blue_survey = BlobSurvey(
            [blue_fp_basis, m5_basis_g],
            [footprint_weight, m5_weight],
            bandname1="g",
            survey_name="g_blob",
            nside=nside,
            camera="comcam",
            grow_blob=False,
            detailers=detailers,
            dither=False,
            twilight_scale=False,
        )

        scheduler = CoreScheduler([red_survey, blue_survey], nside=nside, camera="comcam")

        mo, scheduler, observations = sim_runner(mo, scheduler, sim_duration=0.5, verbose=True)

        assert len(observations) > 100
        assert np.size(np.where(observations["band"] == "r")[0]) > 0
        assert np.size(np.where(observations["band"] == "g")[0]) > 0


if __name__ == "__main__":
    unittest.main()
