import unittest
import warnings

import numpy as np

import rubin_scheduler.scheduler.basis_functions as basis_functions
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.utils import ObservationArray


class TestBasis(unittest.TestCase):
    def setUp(self) -> None:
        self.observatory = ModelObservatory()
        self.conditions = self.observatory.return_conditions()

    def test_basics(self):
        """Test the basics of each basis function"""

        # List of simple basis functions that don't have
        # any required args
        bfs = [
            basis_functions.ConstantBasisFunction,
            basis_functions.DelayStartBasisFunction,
            basis_functions.AvoidFastRevisitsBasisFunction,
            basis_functions.VisitRepeatBasisFunction,
            basis_functions.M5DiffBasisFunction,
            basis_functions.StrictBandBasisFunction,
            basis_functions.BandChangeBasisFunction,
            basis_functions.SlewtimeBasisFunction,
            basis_functions.AzModuloBasisFunction,
            basis_functions.DecModuloBasisFunction,
            basis_functions.NearSunHighAirmassBasisFunction,
            basis_functions.EclipticBasisFunction,
            basis_functions.NGoodSeeingBasisFunction,
            basis_functions.AvoidDirectWind,
            basis_functions.BandDistBasisFunction,
            basis_functions.RewardRisingBasisFunction,
            basis_functions.BandLoadedBasisFunction,
            basis_functions.OnceInNightBasisFunction,
            basis_functions.SunAltHighLimitBasisFunction,
            basis_functions.TimeToTwilightBasisFunction,
            basis_functions.NotTwilightBasisFunction,
            basis_functions.AfterEveningTwiBasisFunction,
            basis_functions.ForceDelayBasisFunction,
            basis_functions.HourAngleLimitBasisFunction,
            basis_functions.MoonDownBasisFunction,
            basis_functions.CloudedOutBasisFunction,
            basis_functions.SoftDelayBasisFunction,
            basis_functions.SunAltLimitBasisFunction,
            basis_functions.NightModuloBasisFunction,
            basis_functions.EndOfEveningBasisFunction,
            basis_functions.TimeToScheduledBasisFunction,
            basis_functions.CloseToTwilightBasisFunction,
            basis_functions.InTimeWindowBasisFunction,
            basis_functions.HaMaskBasisFunction,
            basis_functions.MoonAvoidanceBasisFunction,
            basis_functions.MapCloudBasisFunction,
            basis_functions.PlanetMaskBasisFunction,
            basis_functions.SolarElongationMaskBasisFunction,
            basis_functions.AltAzShadowMaskBasisFunction,
            basis_functions.AltAzShadowTimeLimitedBasisFunction,
            basis_functions.MoonAltLimitBasisFunction,
            basis_functions.RevHaMaskBasisFunction,
            basis_functions.MaskAllButNES,
            basis_functions.NInNightMaskBasisFunction,
            basis_functions.OnlyBeforeNightBasisFunction,
            basis_functions.MaskAfterNObsBasisFunction,
        ]

        obs = ObservationArray()
        obs["band"] = "r"
        obs["mjd"] = 59000.0
        indx = np.array([1000])

        for bf in bfs:
            awake_bf = bf()
            awake_bf.add_observation(obs, indx=indx)
            feas = awake_bf.check_feasibility(conditions=self.conditions)
            reward = awake_bf(self.conditions)
            assert feas is not None
            assert reward is not None

    def test_altazshadowtimelimited(self):
        cond = Conditions()
        bf = basis_functions.AltAzShadowTimeLimitedBasisFunction()

        # Set conditions to be in the middle of the night,
        # bf should return 0
        cond.mjd = 5900
        # Sunrise is in 4 hours
        cond.sunrise = 5900 + 4.0 / 24.0
        cond.sunset = 5900 - 4.0 / 24.0

        assert bf(cond) == 0

        # set to be close to sunrise/set, should return mask
        cond.sunrise = 5900 + 1.0 / 24.0

        mask1 = bf(cond)
        assert np.size(mask1) > 1

        # Make a more restrictive mask
        bf = basis_functions.AltAzShadowTimeLimitedBasisFunction(min_alt=40, max_alt=50)
        mask2 = bf(cond)

        # New mask should have fewer valid pixels than default values
        assert np.size(mask2) > 2
        assert np.sum(np.isfinite(mask1)) > np.sum(np.isfinite(mask2))

    def test_slewtime_basis_function(self):
        current_band = self.conditions.current_filter
        # Test value with matching bandpass
        bf = basis_functions.SlewtimeBasisFunction(bandname=current_band)
        reward = bf(self.conditions)
        # Slewtime reward is always < 0 ..
        self.assertTrue(np.nanmax(reward) < 0)
        # Test value with different bandpass
        bands = "ugrizy"
        different_band = (bands.index(current_band) + 2) % 5
        different_band = bands[different_band]
        bf = basis_functions.SlewtimeBasisFunction(bandname=different_band)
        reward = bf(self.conditions)
        self.assertTrue(np.nanmax(reward) == 0)
        # Test value with bandpass = None
        bf = basis_functions.SlewtimeBasisFunction(bandname=None)
        reward = bf(self.conditions)
        self.assertTrue(np.nanmax(reward) < 0)

    def test_visit_repeat_basis_function(self):
        bf = basis_functions.VisitRepeatBasisFunction()

        indx = np.array([1000])

        # 30 minute step
        delta = 30.0 / 60.0 / 24.0

        # Add 1st observation, should still be zero
        obs = ObservationArray()
        obs["band"] = "r"
        obs["mjd"] = 59000.0
        conditions = Conditions()
        conditions.mjd = np.max(obs["mjd"])
        bf.add_observation(obs, indx=indx)
        self.assertEqual(np.max(bf(conditions)), 0.0)

        # Advance time so now we want a pair
        conditions.mjd += delta
        self.assertEqual(np.max(bf(conditions)), 1.0)

        # Now complete the pair and it should go back to zero
        bf.add_observation(obs, indx=indx)

        conditions.mjd += delta
        self.assertEqual(np.max(bf(conditions)), 0.0)

    def test_force_delay(self):
        bf = basis_functions.ForceDelayBasisFunction(days_delay=3.0, scheduler_note="survey")
        obs = ObservationArray()
        obs["scheduler_note"] = "not_match"
        obs["mjd"] = 10
        bf.add_observation(obs)
        conditions = Conditions()
        conditions.mjd = 11.0
        assert bf.check_feasibility(conditions)
        # Now it matches, so should block
        obs["scheduler_note"] = "survey"
        bf.add_observation(obs)

        assert not bf.check_feasibility(conditions)

    def test_label(self):
        bf = basis_functions.VisitRepeatBasisFunction()
        self.assertIsInstance(bf.label(), str)

        bf = basis_functions.SlewtimeBasisFunction(nside=16)
        self.assertIsInstance(bf.label(), str)

    def test_visit_gap(self):
        visit_gap = basis_functions.VisitGap(note="test")

        conditions = Conditions()
        conditions.mjd = 59000.0

        # default is feasible
        assert visit_gap.check_feasibility(conditions=conditions)

        observation = ObservationArray()
        observation["band"] = "r"
        observation["scheduler_note"] = "foo"
        observation["mjd"] = 59000.0

        visit_gap.add_observation(observation=observation)

        # observation with the wrong note
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["scheduler_note"] = "test"
        visit_gap.add_observation(observation=observation)

        # now observation with the correct note
        assert not visit_gap.check_feasibility(conditions=conditions)

        # check it becomes feasible again once enough time has passed
        conditions.mjd += 2.0 * visit_gap.gap

        assert visit_gap.check_feasibility(conditions=conditions)

    def test_visit_gap_with_band(self):
        visit_gap = basis_functions.VisitGap(note="test", band_names=["g"])

        conditions = Conditions()
        conditions.mjd = 59000.0

        # default is feasible
        assert visit_gap.check_feasibility(conditions=conditions)

        observation = ObservationArray()
        observation["band"] = "r"
        observation["scheduler_note"] = "foo"
        observation["mjd"] = 59000.0

        visit_gap.add_observation(observation=observation)

        # observation with the wrong note
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["scheduler_note"] = "test"
        visit_gap.add_observation(observation=observation)

        # observation with the wrong band
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["band"] = "g"
        visit_gap.add_observation(observation=observation)

        # observation with the correct note and band
        assert not visit_gap.check_feasibility(conditions=conditions)

        # check it becomes feasible again once enough time has passed
        conditions.mjd += 2.0 * visit_gap.gap

        assert visit_gap.check_feasibility(conditions=conditions)

    def test_visit_gap_with_multiple_bands(self):
        visit_gap = basis_functions.VisitGap(note="test", band_names=["g", "i"])

        conditions = Conditions()
        conditions.mjd = 59000.0

        # default is feasible
        assert visit_gap.check_feasibility(conditions=conditions)

        observation = ObservationArray()
        observation["band"] = "r"
        observation["scheduler_note"] = "foo"
        observation["mjd"] = 59000.0

        visit_gap.add_observation(observation=observation)

        # observation with the wrong note
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["scheduler_note"] = "test"
        visit_gap.add_observation(observation=observation)

        # observation with the wrong band
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["band"] = "g"
        observation["mjd"] += 1e-3
        visit_gap.add_observation(observation=observation)

        # observation with the correct note but only one band
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["band"] = "i"
        observation["mjd"] += 1e-3
        visit_gap.add_observation(observation=observation)

        # observation with the correct note and both bands
        assert not visit_gap.check_feasibility(conditions=conditions)

        # make sure it is still not feasible after only the g observation gap
        # has passed
        conditions.mjd += visit_gap.gap + 1.1e-3

        # observation with the correct note and both bands
        assert not visit_gap.check_feasibility(conditions=conditions)

        # make sure it is feasible after both gaps have passed
        conditions.mjd += 1e-3

        assert visit_gap.check_feasibility(conditions=conditions)

    def test_sun_alt(self):
        sunaltbf = basis_functions.SunAltHighLimitBasisFunction(alt_limit=-15)
        conditions = Conditions()
        conditions.sun_alt = np.radians(-20)
        assert not sunaltbf.check_feasibility(conditions)
        conditions.sun_alt = np.radians(-10)
        assert sunaltbf.check_feasibility(conditions)

    def test_close_to_twilight(self):
        bf = basis_functions.CloseToTwilightBasisFunction(
            max_sun_alt_limit=-14.8, max_time_to_12deg=21.0, min_time_remaining=15.0
        )
        conditions = Conditions()
        conditions.mjd = 520900.00
        conditions.sun_alt = np.radians(-14)
        conditions.sun_n12_rising = conditions.mjd + 16.0 / 60 / 24
        assert bf.check_feasibility(conditions)
        conditions.sun_n12_rising = conditions.mjd + 14.0 / 60 / 24
        assert not bf.check_feasibility(conditions)
        conditions.mjd = 520900.00
        conditions.sun_n12_rising = conditions.mjd + 16.0 / 60 / 24
        conditions.sun_alt = np.radians(-20)
        assert bf.check_feasibility(conditions)

    def test_AltAzShadowMask(self):
        nside = 32
        conditions = Conditions(nside=nside)
        conditions.mjd = 59000.0
        conditions.tel_az_limits = [np.radians(-250), np.radians(250)]
        conditions.tel_alt_limits = [np.radians(-100), np.radians(100)]
        # With no (real) limits, including no limits in conditions
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=0, max_alt=90, min_az=0, max_az=360, shadow_minutes=0, pad=0
        )
        result = bf(conditions)
        self.assertTrue(np.all(np.isnan(result[np.where(conditions.alt < 0)])))
        self.assertTrue(np.all(result[np.where(conditions.alt >= 0)] == 0))
        # Set altitude limits but still no padding
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=40, max_alt=60, min_az=0, max_az=360, shadow_minutes=0, pad=0
        )
        result = bf(conditions)
        good = np.where((conditions.alt > np.radians(40)) & (conditions.alt < np.radians(60)), True, False)
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # And set azimuth limits but no padding
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=100, min_az=90, max_az=180, shadow_minutes=0, pad=0
        )
        result = bf(conditions)
        good = np.where((conditions.az > np.radians(90)) & (conditions.az < np.radians(180)), True, False)
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # And set azimuth limits - order sensitive
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=180, max_az=90, shadow_minutes=0, pad=0
        )
        result = bf(conditions)
        good = np.where((conditions.az > np.radians(180)) | (conditions.az < np.radians(90)), True, False)
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # And set altitude limits from the conditions
        conditions.sky_alt_limits = [[np.radians(40), np.radians(60)]]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=0, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where((conditions.alt > np.radians(43)) & (conditions.alt < np.radians(57)))
        bad = np.where((conditions.alt < np.radians(42)) & (conditions.alt > np.radians(58)))
        self.assertTrue(np.all(np.isnan(result[bad])))
        self.assertTrue(np.all(result[good] == 0))
        # Set multiple altitude limits from the conditions
        conditions.sky_alt_limits = [[np.radians(40), np.radians(60)], [np.radians(80), np.radians(90)]]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0, pad=3
        )
        result = bf(conditions)
        # Conditions value get padded
        good = np.where(
            ((conditions.alt > np.radians(43)) & (conditions.alt < np.radians(57)))
            | ((conditions.alt > np.radians(83)) & (conditions.alt < np.radians(87))),
        )
        bad = np.where(
            (conditions.alt < np.radians(41))
            | ((conditions.alt > np.radians(59)) & (conditions.alt < np.radians(81)))
        )
        self.assertTrue(np.all(np.isnan(result[bad])))
        self.assertTrue(np.all(result[good] == 0))
        # Set azimuth limits
        conditions.sky_alt_limits = None
        conditions.sky_az_limits = [[np.radians(270), np.radians(90)]]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        # Conditions value get padded, but we should be able to mask this area
        # at a minimum
        good = np.where(
            (conditions.az > np.radians(270)) | (conditions.az < np.radians(90)),
            True,
            False,
        )
        self.assertTrue(np.all(np.isnan(result[~good])))
        # Set azimuth limits, direction sensitive
        conditions.sky_alt_limits = None
        conditions.sky_az_limits = [[np.radians(90), np.radians(270)]]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0, pad=2
        )
        result = bf(conditions)
        # To more accurately count in azimuth, remove very high altitudes
        good = np.where(
            (conditions.alt < np.radians(80))
            & (conditions.alt > np.radians(0))
            & (conditions.az < np.radians(260))
            & (conditions.az > np.radians(110))
        )
        bad = np.where((conditions.az > np.radians(269)) | (conditions.az < np.radians(91)))
        # Check this area is masked
        self.assertTrue(np.all(np.isnan(result[bad])))
        self.assertTrue(np.all(result[good] == 0))
        # Set altitude limits from kinematic model
        conditions.sky_alt_limits = None
        conditions.tel_alt_limits = [np.radians(20), np.radians(86.5)]
        conditions.sky_az_limits = None
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where(
            (conditions.alt > np.radians(23)) & (conditions.alt < np.radians(84)),
        )
        bad = np.where((conditions.alt < np.radians(21)) | (conditions.alt > np.radians(85)))
        # Check this area is masked
        self.assertTrue(np.all(np.isnan(result[bad])))
        self.assertTrue(np.all(result[good] == 0))
        # Set azimuth limits from kinematic model
        conditions.sky_alt_limits = None
        conditions.sky_az_limits = None
        conditions.tel_az_limits = [np.radians(90), np.radians(270)]
        conditions.tel_alt_limits = [np.radians(-100), np.radians(100)]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        bad = np.where(
            (conditions.az > np.radians(268)) | (conditions.az < np.radians(92)),
        )
        good = np.where(
            (conditions.alt < np.radians(80))
            & (conditions.alt > np.radians(0))
            & (conditions.az < np.radians(260))
            & (conditions.az > np.radians(110))
        )
        # Check this area is masked
        self.assertTrue(np.all(np.isnan(result[bad])))
        self.assertTrue(np.all(result[good] == 0))
        # Check very simple shadow minutes
        # Set azimuth limits from kinematic model
        conditions.sky_alt_limits = None
        conditions.sky_az_limits = None
        conditions.tel_az_limits = [np.radians(-250), np.radians(250)]
        conditions.tel_alt_limits = [np.radians(-100), np.radians(100)]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=20, max_alt=85, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=20, max_alt=85, min_az=0, max_az=360, shadow_minutes=30
        )
        result_shadow = bf(conditions)
        overlap = np.where(np.isnan(result) & ~np.isnan(result_shadow))
        self.assertTrue(len(overlap[0]) == 0)
        overlap = np.where(np.isnan(result_shadow) & ~np.isnan(result))
        self.assertTrue(len(overlap[0]) > 0)

        # Check that shadow is being computed multiple times
        # basis function where time step is too large, so zenith exclusion path
        # will not be filled properly
        bf_gap = basis_functions.AltAzShadowMaskBasisFunction(shadow_minutes=120.0, time_step=500.0)
        # Better time_step so path of zenith is masked. So bf_filled should
        # have fewer valid pixels.
        bf_filled = basis_functions.AltAzShadowMaskBasisFunction(shadow_minutes=120.0, time_step=10.0)

        mask_gap = bf_gap(conditions)
        mask_filled = bf_filled(conditions)

        assert np.sum(np.isfinite(mask_gap)) > np.sum(np.isfinite(mask_filled))

    def test_deprecated(self):
        # Add to-be-deprecated functions here as they appear
        deprecated_basis_functions = [
            basis_functions.SolarElongMaskBasisFunction,
            basis_functions.AzimuthBasisFunction,
            basis_functions.SeasonCoverageBasisFunction,
            basis_functions.TimeInTwilightBasisFunction,
            basis_functions.FilterLoadedBasisFunction,
            basis_functions.StrictFilterBasisFunction,
            basis_functions.FilterChangeBasisFunction,
            basis_functions.FilterDistBasisFunction,
        ]
        for dep_bf in deprecated_basis_functions:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dep_bf()
                # Verify deprecation warning
                assert len(w) >= 1
                assert issubclass(w[-1].category, (DeprecationWarning, FutureWarning))


if __name__ == "__main__":
    unittest.main()
