import copy
import unittest

import numpy as np

import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.utils import CurrentAreaMap, ObservationArray, TargetoO
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
            detailers.CameraRotDetailer,
            detailers.CameraSmallRotPerObservationListDetailer,
            detailers.BandToFilterDetailer,
            detailers.TagRadialDetailer,
            detailers.LabelDDFDetailer,
            detailers.TruncatePreTwiDetailer,
        ]

        for det in det_list:
            obs = copy.deepcopy(obs_array)
            live_det = det()
            result = live_det(obs, conditions)
            assert len(result) > 0

        # test the Euclid detailer
        live_det = detailers.EuclidDitherDetailer()
        # No "EDFS" should raise an error
        with self.assertRaises(ValueError):
            obs = copy.deepcopy(obs_array)
            live_det(obs, conditions)

        # Update so detailer should run
        obs = copy.deepcopy(obs_array)
        obs["scheduler_note"] = "DD:EDFS_a, 1212, 12, abc"
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

        conditions = Conditions(mjd=3, survey_start_mjd=0)
        conditions.mounted_bands = ["i", "z"]

        out_obs = det(obs, conditions)
        assert (out_obs["band"] == "i") | (out_obs["band"] == "z")

        # Check that we fall back properly
        conditions.mounted_bands = ["r", "g", "u", "y"]
        det = detailers.RandomBandDetailer(bands="iz", fallback_order="y")
        out_obs = det(obs, conditions)

        assert out_obs["band"] == "y"

    def test_band_sort_detailer(self):
        obs = ObservationArray(5)
        input_bands = ["r", "g", "r", "g", "z"]
        for i in range(len(obs)):
            obs[i]["band"] = input_bands[i]

        conditions = Conditions()
        conditions.mounted_bands = ["g", "r", "i", "z", "y"]
        conditions.current_band = "g"

        det = detailers.BandSortDetailer(desired_band_order="rgz", loaded_first=True)
        out_obs = det(obs, conditions)
        self.assertTrue(np.all(out_obs["band"] == ["g", "g", "r", "r", "z"]))

        det = detailers.BandSortDetailer(desired_band_order="rgz", loaded_first=False)
        out_obs = det(obs, conditions)
        self.assertTrue(np.all(out_obs["band"] == ["r", "r", "g", "g", "z"]))

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

    def test_copycol(self):
        obs = ObservationArray(3)
        obs["band"][0] = "r"
        obs["band"][1] = "g"
        obs["band"][2] = "g"

        detailer = detailers.CopyValueDetailer("band", "filter")

        output = detailer(obs, None)

        assert np.array_equal(output["band"], output["filter"])

    def test_event(self):
        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        detailer = detailers.BandPickToODetailer(
            band_start="g", band_end="r", distance_limit=10.0, check_mounted=True
        )

        obs = ObservationArray(3)
        obs["RA"] = np.radians(0.0)
        obs["dec"] = np.radians(-20.0)
        obs["band"] = "g"

        too_event = TargetoO(
            100, None, 10, 10, ra_rad_center=None, dec_rad_center=None, too_type=None, posterior_distance=1e9
        )

        # Should not do anything if distance to large
        out_obs = detailer(obs, conditions, target_o_o=too_event)
        assert np.all(out_obs["band"] == "g")

        # Now it is small enough
        too_event.posterior_distance = 0
        out_obs = detailer(obs, conditions, target_o_o=too_event)
        assert np.all(out_obs["band"] == "r")

        # reset
        obs["band"] = "g"

        # does nothing if not mounted
        conditions.mounted_bands = ["a", "b", "c"]
        out_obs = detailer(obs, conditions, target_o_o=too_event)
        assert np.all(out_obs["band"] == "g")

        # Don't care if mounted, swap it
        detailer = detailers.BandPickToODetailer(
            band_start="g", band_end="r", distance_limit=10.0, check_mounted=False
        )

        out_obs = detailer(obs, conditions, target_o_o=too_event)
        assert np.all(out_obs["band"] == "r")

    def test_trackinginfo(self):
        obs = ObservationArray(2)
        conditions = Conditions()

        # Check basic function
        det = detailers.TrackingInfoDetailer(
            target_name="", observation_reason="pairs_12", science_program="BLOCK-T"
        )

        out_obs = det(obs, conditions)
        for o in out_obs:
            self.assertEqual(o["observation_reason"], "pairs_12")
            self.assertEqual(o["science_program"], "BLOCK-T")
            self.assertEqual(o["target_name"], "")

        # Check that it replaces spaces with underscores
        obs = ObservationArray(2)
        det = detailers.TrackingInfoDetailer(
            target_name="", observation_reason="pairs space", science_program="BLOCK-T"
        )

        out_obs = det(obs, conditions)
        for o in out_obs:
            self.assertEqual(o["observation_reason"], "pairs_space")
            self.assertEqual(o["science_program"], "BLOCK-T")
            self.assertEqual(o["target_name"], "")

        # Check that it doesn't overwrite existing data
        det = detailers.TrackingInfoDetailer(
            target_name="", observation_reason="pairs_12", science_program="BLOCK-T"
        )
        out_obs = det(out_obs, conditions)
        for o in out_obs:
            self.assertEqual(o["observation_reason"], "pairs_space")

    def test_band_sort(self):
        """Test BandSortDetailer"""

        detailer = detailers.BandSortDetailer()
        conditions = Conditions()

        conditions.current_band = "i"

        obs = ObservationArray(2)
        obs["band"][0] = "r"
        obs["band"][1] = "z"

        # Default settings should not change order
        obs_out = detailer(obs, conditions)
        assert (obs_out["band"][0] == "r") & (obs_out["band"][1] == "z")

        # Swap, make sure still doesn't change order
        obs = ObservationArray(2)
        obs["band"][0] = "z"
        obs["band"][1] = "r"

        # Default settings should not change order
        obs_out = detailer(obs, conditions)
        assert (obs_out["band"][1] == "r") & (obs_out["band"][0] == "z")

        # If r is loaded, should always swap to having r first
        conditions.current_band = "r"

        obs = ObservationArray(2)
        obs["band"][0] = "r"
        obs["band"][1] = "z"

        # Default settings should not change order
        obs_out = detailer(obs, conditions)
        assert (obs_out["band"][0] == "r") & (obs_out["band"][1] == "z")

        # Swap, make sure still doesn't change order
        obs = ObservationArray(2)
        obs["band"][0] = "z"
        obs["band"][1] = "r"

        # Default settings should not change order
        obs_out = detailer(obs, conditions)
        assert (obs_out["band"][0] == "r") & (obs_out["band"][1] == "z")

        # If we want a specified order, ignore what's loaded
        detailer = detailers.BandSortDetailer(desired_band_order="yzi", loaded_first=False)
        conditions = Conditions()
        conditions.current_band = "i"
        obs = ObservationArray(3)
        obs["band"][0] = "i"
        obs["band"][1] = "y"
        obs["band"][2] = "z"

        obs_out = detailer(obs, conditions)

        assert obs_out["band"][0] == "y"
        assert obs_out["band"][1] == "z"
        assert obs_out["band"][2] == "i"

        # Same, but now do bump up band if loaded
        detailer = detailers.BandSortDetailer(desired_band_order="yzi", loaded_first=True)
        conditions = Conditions()
        conditions.current_band = "i"
        obs = ObservationArray(3)
        obs["band"][0] = "i"
        obs["band"][1] = "y"
        obs["band"][2] = "z"

        obs_out = detailer(obs, conditions)

        assert obs_out["band"][0] == "i"
        assert obs_out["band"][1] == "y"
        assert obs_out["band"][2] == "z"

    def test_splitdither(self):
        obs = ObservationArray(5)
        input_notes = ["survey_a", "survey_b", "survey_a", "survey_b", "survey_a"]
        for i in range(len(obs)):
            obs[i]["scheduler_note"] = input_notes[i]

        conditions = Conditions()

        # This is more likely used to split between dither patterns,
        # But for easy test let's just assign different target_names.
        det_a = detailers.TrackingInfoDetailer(target_name="a")
        det_b = detailers.TrackingInfoDetailer(target_name="b")

        det = detailers.SplitDetailer(det_a, det_b, split_str="_b")
        out_obs = det(obs, conditions)

        for o in out_obs:
            assert o["scheduler_note"].split("_")[-1] == o["target_name"]

    def test_region_label(self):

        for detailer_start in [detailers.LabelRegionDetailer, detailers.LabelRegionsAndDDFs]:
            nside = 32
            sky = CurrentAreaMap(nside=nside)
            footprints, labels = sky.return_maps()

            obs = ObservationArray(3)
            obs["RA"] = np.radians(0.0)
            obs["dec"] = np.radians(-20.0)

            obs["target_name"] = "dummy_start"

            detailer = detailer_start(label_array=labels)

            output = detailer(obs, None)
            for res in output:
                assert "dummy" in res["target_name"]
                assert "lowdust" in res["target_name"]

        # Test that we clobber
        detailer = detailers.LabelRegionDetailer(labels, append=False)
        output = detailer(obs, None)
        for res in output:
            assert "dummy" not in res["target_name"]
            assert "lowdust" in res["target_name"]

        # Test that we can have multiple labels
        obs = ObservationArray(3)
        obs["RA"] = np.radians(0.0)
        obs["dec"] = np.radians(3.0)
        detailer = detailers.LabelRegionDetailer(labels)
        output = detailer(obs, None)
        for res in output:
            assert "dummy" not in res["target_name"]
            assert "lowdust" in res["target_name"]
            assert "nes" in res["target_name"]

    def test_euclid_dither(self):
        detailer = detailers.EuclidDitherDetailer(per_night=True)

        ra_a, dec_a, ra_b, dec_b = detailer._generate_offsets(100, 2)

        # Going per night, should only be one offset
        assert np.size(ra_a) == 1

        # if we call again, should be the same
        ra_ap, dec_ap, ra_bp, dec_bp = detailer._generate_offsets(100, 2)

        assert ra_a == ra_ap
        assert dec_a == dec_ap
        assert ra_b == ra_bp
        assert dec_b == dec_bp

        # Now want a unique position
        detailer = detailers.EuclidDitherDetailer(per_night=False)

        ra_a, dec_a, ra_b, dec_b = detailer._generate_offsets(100, 2)

        assert np.size(ra_a) == 100

        # If those get executed, they get added to the detailer
        obs = ObservationArray(n=100)
        obs["night"] = 2
        detailer.add_observations_array(obs, None)

        # call again should result in different values
        ra_ap, dec_ap, ra_bp, dec_bp = detailer._generate_offsets(100, 2)
        assert np.all(np.not_equal(ra_a, ra_ap))

    def test_dither(self):
        detailer = detailers.DitherDetailer(per_night=True)

        ra_a, dec_a = detailer._generate_offsets(100, 2)
        # Going per night, should only be one offset
        assert np.size(np.unique(ra_a)) == 1

        # if we call again, should be the same
        ra_ap, dec_ap = detailer._generate_offsets(100, 2)

        assert np.all(ra_a == ra_ap)
        assert np.all(dec_a == dec_ap)

        # Now want a unique position
        detailer = detailers.DitherDetailer(per_night=False)

        ra_a, dec_a = detailer._generate_offsets(100, 2)

        assert np.size(np.unique(ra_a)) == 100

        # If those get executed, they get added to the detailer
        obs = ObservationArray(n=100)
        obs["night"] = 2
        detailer.add_observations_array(obs, None)

        # call again should result in different values
        ra_ap, dec_ap = detailer._generate_offsets(100, 2)

        assert np.all(np.not_equal(ra_a, ra_ap))

        # Check that negative nights are ok
        detailer = detailers.DitherDetailer(per_night=True)
        detailer._new_ang_rad(-10)
        ang1 = detailer.angles.copy()
        detailer._new_ang_rad(10)
        ang2 = detailer.angles.copy()

        assert np.all(ang1 != ang2)

    def test_truncate(self):
        """Test we truncate observations before twilight"""
        detailer = detailers.TruncatePreTwiDetailer()
        observation_array = ObservationArray(n=100)
        n1 = np.size(observation_array)

        observation_array["band"] = "r"
        observation_array["exptime"] = 30.0

        conditions = Conditions(mjd=3, survey_start_mjd=0)
        conditions.current_band = "r"
        # Have a ton of time, no obs should be cut
        conditions.sun_n18_rising = 3.5
        obs_out = detailer(observation_array, conditions)

        assert np.size(obs_out) == n1

        # cut down to not enough time
        conditions.sun_n18_rising = 3.016

        obs_out = detailer(observation_array, conditions)
        n2 = np.size(obs_out)

        # Now if we are in a different filter, should
        # Have to lose a few more, but not all
        conditions.current_band = "b"
        obs_out = detailer(observation_array, conditions)

        n3 = np.size(obs_out)

        assert n1 > n2

        assert n2 > n3

        assert n3 > 0

    def test_rollband(self):

        orig_order = np.array(["u", "g", "r"])

        observation_array = ObservationArray(n=3)
        observation_array["band"] = ["u", "g", "r"]

        # Test where no change should happen
        detailer = detailers.RollBandMatchDetailer()
        conditions = Conditions()
        conditions.current_band = "u"
        o1 = detailer(observation_array, conditions)
        for bn, ob in zip(orig_order, o1["band"]):
            assert bn == ob

        # Test where things should roll
        conditions.current_band = "r"
        o2 = detailer(observation_array, conditions)

        assert o2["band"][0] == "r"

        for bn in orig_order:
            assert bn in o2["band"]


if __name__ == "__main__":
    unittest.main()
