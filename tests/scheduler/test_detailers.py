import copy
import unittest

import numpy as np

import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.utils import ObservationArray
from rubin_scheduler.utils import DEFAULT_NSIDE, _ra_dec2_hpid


class TestDetailers(unittest.TestCase):
    def test_basics(self):
        """Test basic detailer functionality"""

        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        dec = np.radians(-20)
        ra = np.arange(0, 2 * np.pi, np.pi / 4)

        obs_array = ObservationArray(n=ra.size)
        obs_array["band"] = "r"
        obs_array["RA"] = ra
        obs_array["dec"] = dec
        obs_array["mjd"] = 59000.0
        obs_array["exptime"] = 30.0
        obs_array["scheduler_note"] = "test_note, a"

        hpid = _ra_dec2_hpid(DEFAULT_NSIDE, ra, dec)

        indx = np.isfinite(conditions.m5_depth["r"][hpid])
        obs_array = obs_array[indx]

        det_list = [
            detailers.VaryExptDetailer,
            detailers.ZeroRotDetailer,
            detailers.Comcam90rotDetailer,
            detailers.Rottep2RotspDesiredDetailer,
            detailers.CloseAltDetailer,
            detailers.TakeAsPairsDetailer,
            detailers.TwilightTripleDetailer,
            detailers.FlushForSchedDetailer,
            detailers.BandNexp,
            detailers.FixedSkyAngleDetailer,
            detailers.ParallacticRotationDetailer,
            detailers.FlushByDetailer,
            detailers.RandomBandDetailer,
            detailers.TrackingInfoDetailer,
            detailers.AltAz2RaDecDetailer,
            detailers.DitherDetailer,
            detailers.EuclidDitherDetailer,
            detailers.CameraRotDetailer,
            detailers.CameraSmallRotPerObservationListDetailer,
            detailers.BandToFilterDetailer,
        ]

        for det in det_list:
            obs = copy.deepcopy(obs_array)
            live_det = det()
            result = live_det(obs, conditions)
            assert len(result) > 0

    def test_start_field(self):
        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        scheduler_note = "prepended"

        obs_to_prepend = ObservationArray(n=3)
        obs_to_prepend["RA"] = np.radians(20)
        obs_to_prepend["band"] = "r"
        obs_to_prepend["scheduler_note"] = scheduler_note

        obs_reg = ObservationArray(n=1)
        det = detailers.StartFieldSequenceDetailer(obs_to_prepend, scheduler_note=scheduler_note)
        obs_out = det(obs_reg, conditions)

        assert len(obs_out) == 4

    def test_delta_dither(self):
        obs = ObservationArray(2)
        obs["RA"] = np.radians(0)
        obs["dec"] = np.radians(20)

        det = detailers.DeltaCoordDitherDetailer(np.array([0.0, 0.0]), np.array([-10.0, 10.0]))

        output = det(obs, [])

        assert np.size(output) == 4
        assert np.unique(output["RA"]) == np.unique(obs["RA"])

        obs["RA"] = np.radians(90)
        det = detailers.DeltaCoordDitherDetailer(np.array([0.0, 0.0]), np.array([-10.0, 10.0]))
        output = det(obs, [])
        assert np.size(output) == 4
        assert np.unique(output["RA"]) == np.unique(obs["RA"])

        ra_step = 5.0
        det = detailers.DeltaCoordDitherDetailer(np.array([-ra_step, ra_step]), np.array([0.0, 0.0]))
        output = det(obs, [])

        # Make sure ra diff is larger
        assert np.all(np.abs(output["RA"] - obs["RA"][0]) > np.radians(ra_step))

        # Try having rotation as well
        det = detailers.DeltaCoordDitherDetailer(
            np.array([-ra_step, ra_step]), np.array([0.0, 0.0]), delta_rotskypos=np.array([5.0, 10.0])
        )
        output = det(obs, [])

        assert np.size(output) == 4
        assert np.size(np.unique(output["rotSkyPos_desired"])) == 2

        # Make sure one-element works
        obs = ObservationArray(1)
        obs["RA"] = np.radians(0)
        obs["dec"] = np.radians(20)
        det = detailers.DeltaCoordDitherDetailer(np.array([0.0, 0.0]), np.array([-10.0, 10.0]))
        output = det(obs, [])

        assert np.size(output) == 2
        assert np.unique(output["RA"]) == np.unique(obs["RA"])

        # Test going to the pole
        obs = ObservationArray(2)
        obs["RA"] = np.radians(0)
        obs["dec"] = np.radians(-90)

        det = detailers.DeltaCoordDitherDetailer(np.array([-1.0, 1.0, -1, 1]), np.array([1.0, 1.0, -1, -1]))
        output = det(obs, [])

        # This should make a grid all at the same dec
        assert np.size(np.unique(output["dec"])) == 1
        assert output["dec"][0] > obs["dec"][0]

    def test_random_band(self):
        obs = ObservationArray(1)
        obs["band"] = "r"

        det = detailers.RandomBandDetailer(bands="iz")

        conditions = Conditions()
        conditions.night = 3
        conditions.mounted_bands = ["i", "z"]

        out_obs = det(obs, conditions)
        assert (out_obs["band"] == "i") | (out_obs["filter"] == "z")

        # Check that we fall back properly
        conditions.mounted_bands = ["r", "g", "u", "y"]
        det = detailers.RandomBandDetailer(bands="iz", fallback_order="y")
        out_obs = det(obs, conditions)

        assert out_obs["band"] == "y"

    def test_bandtofilter(self):
        obs = ObservationArray(3)
        obs["band"][0] = "r"
        obs["band"][1] = "g"
        obs["band"][2] = "g"

        detailer = detailers.BandToFilterDetailer({})
        output = detailer(obs, [])
        assert np.all(output["band"] == output["filter"])

        filtername_dict = {"r": "r_03", "g": "g_01"}
        detailer = detailers.BandToFilterDetailer(filtername_dict)
        output = detailer(obs, [])

        for out in output:
            assert out["filter"] == filtername_dict[out["band"]]


if __name__ == "__main__":
    unittest.main()
