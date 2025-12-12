import unittest

import healpy as hp
import numpy as np

import rubin_scheduler.scheduler.basis_functions as bf
from rubin_scheduler.scheduler.detailers import RotspUpdateDetailer
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.schedulers import BaseQueueManager
from rubin_scheduler.scheduler.utils import Footprint, ObservationArray
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD, hpid2_ra_dec


class TestQM(unittest.TestCase):

    def test_qm(self):
        observations = ObservationArray(n=3)
        observations["RA"] = np.radians(np.arange(0, 3, 90))
        observations["dec"] = np.radians([-20, -30, -40])
        observations["band"] = "r"
        observations["target_id"] = np.arange(3)

        conditions = Conditions()
        conditions.mjd = SURVEY_START_MJD + 10

        detailer = RotspUpdateDetailer()

        # make a footprint, mask it off below dec -30
        npix = hp.nside2npix(DEFAULT_NSIDE)
        fp = np.ones(hp.nside2npix(DEFAULT_NSIDE))
        ra, dec = hpid2_ra_dec(DEFAULT_NSIDE, np.arange(npix))
        fp[np.where(dec < -35)] = 0
        fp_obj = Footprint(SURVEY_START_MJD)
        fp_obj.set_footprint("r", fp)

        fp_bf = bf.FootprintBasisFunction(
            bandname="r",
            footprint=fp_obj,
            out_of_bounds_val=np.nan,
            nside=DEFAULT_NSIDE,
        )

        qm = BaseQueueManager(detailers=[detailer], basis_functions=[fp_bf])

        qm.set_queue(observations)

        # Do we get the active queue?
        active_queue = qm.return_active_queue()
        assert active_queue.size == 3

        one_obs = qm.request_observation(conditions)

        assert np.size(one_obs) == 1

        # Active queue should be the same
        active_queue = qm.return_active_queue()
        assert active_queue.size == 3

        qm.add_observation(one_obs)

        # Now should be down to 2
        active_queue = qm.return_active_queue()
        assert active_queue.size == 2

        qm.set_queue(observations)

        # Active queue should be reset
        active_queue = qm.return_active_queue()
        assert active_queue.size == 3

        # Does no check for visibility
        all_valid_obs = qm.request_observation(conditions, whole_queue=True)
        assert np.size(all_valid_obs) == 3

        # Does check that should toss one for being out of footprint
        all_valid_obs = qm.request_observation(conditions, n_return=4)
        assert np.size(all_valid_obs) == 2

        # Check we can wipe out lots at once
        names = list(observations.dtype.names)
        types = [observations[name].dtype for name in names]
        names.append("hpid")
        types.append(int)
        ndt = list(zip(names, types))
        observations_hpid_in = np.zeros(3, dtype=ndt)
        qm.add_observations_array(observations, observations_hpid_in)
        # Should have matched everything
        active_queue = qm.return_active_queue()
        assert active_queue.size == 0

        qm.set_queue(observations)
        qm.flush_queue()
        active_queue = qm.return_active_queue()
        assert active_queue.size == 0

        # Test the detailer is doing something.
        # shouldn't be able to observe rotSkyPos
        # of 0 and 180, so at least one of these
        # should change.
        qm.set_queue(observations)
        result1 = qm.request_observation(conditions, n_return=2)

        rot_same_1 = np.all(result1["rotSkyPos"] == 0)

        observations["rotSkyPos"] = np.pi
        qm.set_queue(observations)
        result2 = qm.request_observation(conditions, n_return=2)

        rot_same_2 = np.all(result2["rotSkyPos"] == np.pi)

        assert rot_same_1 * rot_same_2 == 0


if __name__ == "__main__":
    unittest.main()
