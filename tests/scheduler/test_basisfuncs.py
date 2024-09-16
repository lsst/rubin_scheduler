import unittest
import warnings

import numpy as np

import rubin_scheduler.scheduler.basis_functions as basis_functions
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.utils import ObservationArray


class TestBasis(unittest.TestCase):
    def test_visit_repeat_basis_function(self):
        bf = basis_functions.VisitRepeatBasisFunction()

        indx = np.array([1000])

        # 30 minute step
        delta = 30.0 / 60.0 / 24.0

        # Add 1st observation, should still be zero
        obs = ObservationArray()
        obs["filter"] = "r"
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

        assert ~bf.check_feasibility(conditions)

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
        observation["filter"] = "r"
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

    def test_visit_gap_with_filter(self):
        visit_gap = basis_functions.VisitGap(note="test", filter_names=["g"])

        conditions = Conditions()
        conditions.mjd = 59000.0

        # default is feasible
        assert visit_gap.check_feasibility(conditions=conditions)

        observation = ObservationArray()
        observation["filter"] = "r"
        observation["scheduler_note"] = "foo"
        observation["mjd"] = 59000.0

        visit_gap.add_observation(observation=observation)

        # observation with the wrong note
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["scheduler_note"] = "test"
        visit_gap.add_observation(observation=observation)

        # observation with the wrong filter
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["filter"] = "g"
        visit_gap.add_observation(observation=observation)

        # observation with the correct note and filter
        assert not visit_gap.check_feasibility(conditions=conditions)

        # check it becomes feasible again once enough time has passed
        conditions.mjd += 2.0 * visit_gap.gap

        assert visit_gap.check_feasibility(conditions=conditions)

    def test_visit_gap_with_multiple_filters(self):
        visit_gap = basis_functions.VisitGap(note="test", filter_names=["g", "i"])

        conditions = Conditions()
        conditions.mjd = 59000.0

        # default is feasible
        assert visit_gap.check_feasibility(conditions=conditions)

        observation = ObservationArray()
        observation["filter"] = "r"
        observation["scheduler_note"] = "foo"
        observation["mjd"] = 59000.0

        visit_gap.add_observation(observation=observation)

        # observation with the wrong note
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["scheduler_note"] = "test"
        visit_gap.add_observation(observation=observation)

        # observation with the wrong filter
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["filter"] = "g"
        observation["mjd"] += 1e-3
        visit_gap.add_observation(observation=observation)

        # observation with the correct note but only one filter
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["filter"] = "i"
        observation["mjd"] += 1e-3
        visit_gap.add_observation(observation=observation)

        # observation with the correct note and both filters
        assert not visit_gap.check_feasibility(conditions=conditions)

        # make sure it is still not feasible after only the g observation gap
        # has passed
        conditions.mjd += visit_gap.gap + 1.1e-3

        # observation with the correct note and both filters
        assert not visit_gap.check_feasibility(conditions=conditions)

        # make sure it is feasible after both gaps have passed
        conditions.mjd += 1e-3

        assert visit_gap.check_feasibility(conditions=conditions)

    def test_sun_alt(self):
        sunaltbf = basis_functions.SunAltHighLimitBasisFunction(alt_limit=-15)
        conditions = Conditions()
        conditions.sun_alt = np.radians(-20)
        assert ~sunaltbf.check_feasibility(conditions)
        conditions.sun_alt = np.radians(-10)
        assert sunaltbf.check_feasibility(conditions)

    def test_close_to_twilight(self):
        bf = basis_functions.CloseToTwilightBasisFunction(
            max_sun_alt_limit=-14.8, max_time_to_12deg=21.0, min_time_remaining=15.0
        )
        conditions = Conditions()
        conditions.mjd = 520900.00
        conditions.sun_alt = -14
        conditions.sun_n12_rising = conditions.mjd + 16.0 / 60 / 24
        assert bf.check_feasibility(conditions)
        conditions.sun_n12_rising = conditions.mjd + 14.0 / 60 / 24
        assert ~bf.check_feasibility(conditions)
        conditions.mjd = 520900.00
        conditions.sun_n12_rising = conditions.mjd + 16.0 / 60 / 24
        conditions.sun_alt = -20
        assert ~bf.check_feasibility(conditions)

    def test_AltAzShadowMask(self):
        nside = 32
        conditions = Conditions(nside=nside)
        conditions.mjd = 59000.0
        conditions.altaz_limit_pad = np.radians(2.0)
        conditions.kinematic_az_limits = [np.radians(-250), np.radians(250)]
        conditions.kinematic_alt_limits = [np.radians(-100), np.radians(100)]
        # With no (real) limits, including no limits in conditions
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=0, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        self.assertTrue(np.all(np.isnan(result[np.where(conditions.alt < 0)])))
        self.assertTrue(np.all(result[np.where(conditions.alt >= 0)] == 0))
        # Set altitude limits
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=40, max_alt=60, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where((conditions.alt > np.radians(40)) & (conditions.alt < np.radians(60)), True, False)
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # And set azimuth limits
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=90, max_az=180, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where((conditions.az > np.radians(90)) & (conditions.az < np.radians(180)), True, False)
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # And set azimuth limits - order sensitive
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=180, max_az=90, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where((conditions.az > np.radians(180)) | (conditions.az < np.radians(90)), True, False)
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # And set altitude limits from the conditions
        conditions.tel_alt_limits = [[np.radians(40), np.radians(60)]]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=0, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where((conditions.alt > np.radians(42)) & (conditions.alt < np.radians(58)), True, False)
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # Set multiple altitude limits from the conditions
        conditions.tel_alt_limits = [[np.radians(40), np.radians(60)], [np.radians(80), np.radians(90)]]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        # Conditions value get padded
        good = np.where(
            ((conditions.alt > np.radians(42)) & (conditions.alt < np.radians(58)))
            | ((conditions.alt > np.radians(82)) & (conditions.alt < np.radians(88))),
            True,
            False,
        )
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # Set azimuth limits
        conditions.tel_alt_limits = None
        conditions.tel_az_limits = [[np.radians(270), np.radians(90)]]
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
        conditions.tel_alt_limits = None
        conditions.tel_az_limits = [[np.radians(90), np.radians(270)]]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where(
            (conditions.az < np.radians(268)) & (conditions.az > np.radians(92)),
            True,
            False,
        )
        # Check this area is masked
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # Set altitude limits from kinematic model
        conditions.tel_alt_limits = None
        conditions.kinematic_alt_limits = [np.radians(20), np.radians(86.5)]
        conditions.tel_az_limits = None
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where(
            (conditions.alt > np.radians(22)) & (conditions.alt < np.radians(84.5)),
            True,
            False,
        )
        # Check this area is masked
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # Set azimuth limits from kinematic model
        conditions.tel_alt_limits = None
        conditions.tel_az_limits = None
        conditions.kinematic_az_limits = [np.radians(90), np.radians(270)]
        conditions.kinematic_alt_limits = [np.radians(-100), np.radians(100)]
        bf = basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=-90, max_alt=90, min_az=0, max_az=360, shadow_minutes=0
        )
        result = bf(conditions)
        good = np.where(
            (conditions.az < np.radians(268)) & (conditions.az > np.radians(92)),
            True,
            False,
        )
        # Check this area is masked
        self.assertTrue(np.all(np.isnan(result[~good])))
        self.assertTrue(np.all(result[good] == 0))
        # Check very simple shadow minutes
        # Set azimuth limits from kinematic model
        conditions.tel_alt_limits = None
        conditions.tel_az_limits = None
        conditions.kinematic_az_limits = [np.radians(-250), np.radians(250)]
        conditions.kinematic_alt_limits = [np.radians(-100), np.radians(100)]
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

    def test_deprecated(self):
        # Add to-be-deprecated functions here as they appear
        deprecated_basis_functions = []
        for dep_bf in deprecated_basis_functions:
            print(dep_bf)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dep_bf()
                # Verify deprecation warning
                assert len(w) >= 1
                assert issubclass(w[-1].category, (DeprecationWarning, FutureWarning))


if __name__ == "__main__":
    unittest.main()
