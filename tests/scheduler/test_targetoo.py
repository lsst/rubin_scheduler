import unittest

import numpy as np

from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.utils import SimTargetooServer, TargetoO
from rubin_scheduler.utils import SURVEY_START_MJD, Site, _approx_ra_dec2_alt_az


def make_conditions(mjd):
    conditions = Conditions(mjd=mjd)
    return conditions


class TestTargetoO(unittest.TestCase):
    def test_init_required_args(self):
        footprint = np.array([1, 0, 1, 0])
        too = TargetoO(tooid="too_1", footprint=footprint, mjd_start=SURVEY_START_MJD)
        np.testing.assert_array_equal(too.footprint, footprint)
        self.assertEqual(too.id, "too_1")
        self.assertEqual(too.mjd_start, SURVEY_START_MJD)
        # Check if default kwargs have remained as expected.
        # But update if reasonable.
        self.assertIsNone(too.duration)
        self.assertIsNone(too.ra_rad_center)
        self.assertIsNone(too.dec_rad_center)
        self.assertIsNone(too.too_type)
        self.assertIsNone(too.posterior_distance)
        self.assertTrue(too.interrupt_queue)

    def test_init_all_args(self):
        footprint = np.array([1, 1, 0, 0])
        too = TargetoO(
            tooid="too_2",
            footprint=footprint,
            mjd_start=SURVEY_START_MJD + 2,
            duration=3.0,
            ra_rad_center=1.0,
            dec_rad_center=-0.5,
            too_type="GW",
            posterior_distance=40.0,
            interrupt_queue=False,
            alt_limit=30,
        )
        self.assertEqual(too.id, "too_2")
        self.assertEqual(too.duration, 3.0)
        self.assertEqual(too.ra_rad_center, 1.0)
        self.assertEqual(too.dec_rad_center, -0.5)
        self.assertEqual(too.too_type, "GW")
        self.assertEqual(too.posterior_distance, 40.0)
        self.assertFalse(too.interrupt_queue)
        self.assertAlmostEqual(too.alt_limit, np.radians(30))

    def test_interrupt_queue_disabled_returns_false(self):
        # Check if queue interrupt is false when kwarg is set False.
        too = TargetoO(
            tooid="too_a",
            footprint=np.array([1, 0]),
            mjd_start=SURVEY_START_MJD,
            interrupt_queue=False,
        )
        self.assertFalse(too.queue_should_flush(make_conditions(mjd=SURVEY_START_MJD)))

    def test_radec_center_above_alt_limit_returns_true(self):
        # Might need to adjust this position with new SURVEY_START_MJD?
        ra_center = 2.0
        dec_center = -0.4
        mjd = SURVEY_START_MJD
        lsst = Site("LSST")
        expected_alt, expected_az = _approx_ra_dec2_alt_az(
            ra_center, dec_center, lsst.latitude_rad, lsst.longitude_rad, mjd=mjd
        )
        too = TargetoO(
            tooid="too_b",
            footprint=np.array([1, 0]),
            mjd_start=mjd,
            ra_rad_center=ra_center,
            dec_rad_center=dec_center,
            alt_limit=np.degrees(expected_alt) - 10,
        )
        self.assertTrue(too.queue_should_flush(make_conditions(mjd=mjd)))

    def test_radec_center_below_alt_limit_returns_false(self):
        ra_center = 2.0
        dec_center = -0.4
        mjd = SURVEY_START_MJD
        lsst = Site("LSST")
        expected_alt, expected_az = _approx_ra_dec2_alt_az(
            ra_center, dec_center, lsst.latitude_rad, lsst.longitude_rad, mjd=mjd
        )
        too = TargetoO(
            tooid="too_c",
            footprint=np.array([1, 0]),
            mjd_start=mjd,
            ra_rad_center=ra_center,
            dec_rad_center=dec_center,
            alt_limit=np.degrees(expected_alt) + 10,
        )
        self.assertFalse(too.queue_should_flush(make_conditions(mjd=mjd)))

    def test_footprint_visible_above_alt_limit_returns_true(self):
        # Make the conditions
        conditions = make_conditions(SURVEY_START_MJD)
        # Then pull out a couple of ra/dec positions for the "footprint"
        # using those altitude limits.
        idx = np.where(conditions.alt > np.radians(30))
        footprint = np.zeros(len(conditions.alt))
        footprint[idx[0:3]] = 1
        too = TargetoO(tooid="too_d", footprint=footprint, mjd_start=SURVEY_START_MJD, alt_limit=30)
        self.assertTrue(too.queue_should_flush(conditions))

    def test_footprint_only_zero_pixels_above_limit_returns_false(self):
        # Make the conditions
        conditions = make_conditions(SURVEY_START_MJD)
        # Then pull out a couple of ra/dec positions for the "footprint"
        # using those altitude limits.
        idx = np.where(conditions.alt > np.radians(30))
        footprint = np.ones(len(conditions.alt))
        footprint[idx] = 0
        too = TargetoO(tooid="too_e", footprint=footprint, mjd_start=SURVEY_START_MJD, alt_limit=35)
        self.assertFalse(too.queue_should_flush(conditions))


def make_too(tooid, mjd_start, duration):
    """Build a real TargetoO with a minimal footprint."""
    return TargetoO(
        tooid=tooid,
        footprint=np.array([1, 0]),
        mjd_start=mjd_start,
        duration=duration,
    )


class TestSimTargetooServer(unittest.TestCase):
    def test_init_computes_starts_and_ends(self):
        toos = [
            make_too("a", mjd_start=SURVEY_START_MJD, duration=2.0),
            make_too("b", mjd_start=SURVEY_START_MJD + 10.0, duration=5.0),
        ]
        server = SimTargetooServer(toos)
        np.testing.assert_array_equal(server.mjd_starts, [SURVEY_START_MJD, SURVEY_START_MJD + 10.0])
        np.testing.assert_array_equal(server.mjd_ends, [SURVEY_START_MJD + 2.0, SURVEY_START_MJD + 15.0])

    def test_call_returns_none_before_any_event(self):
        toos = [make_too("a", mjd_start=SURVEY_START_MJD, duration=2.0)]
        server = SimTargetooServer(toos)
        self.assertIsNone(server(SURVEY_START_MJD - 1.0))

    def test_call_returns_none_after_all_events(self):
        toos = [make_too("a", mjd_start=SURVEY_START_MJD, duration=2.0)]
        server = SimTargetooServer(toos)
        self.assertIsNone(server(SURVEY_START_MJD + 3.0))

    def test_call_returns_active_event(self):
        toos = [make_too("a", mjd_start=SURVEY_START_MJD, duration=2.0)]
        server = SimTargetooServer(toos)
        result = server(SURVEY_START_MJD + 1.0)
        self.assertIsNotNone(result)
        self.assertEqual([too.id for too in result], ["a"])

    def test_call_returns_multiple_overlapping_events(self):
        toos = [
            make_too("a", mjd_start=SURVEY_START_MJD, duration=5.0),  # +0 .. +5
            make_too("b", mjd_start=SURVEY_START_MJD + 2.0, duration=5.0),  # +2 .. +7
        ]
        server = SimTargetooServer(toos)
        result = server(SURVEY_START_MJD + 3.0)  # both active
        self.assertEqual([too.id for too in result], ["a", "b"])

    def test_call_returns_only_currently_active_event(self):
        toos = [
            make_too("a", mjd_start=SURVEY_START_MJD, duration=2.0),  # +0 .. +2
            make_too("b", mjd_start=SURVEY_START_MJD + 10.0, duration=2.0),  # +10 .. +12
        ]
        server = SimTargetooServer(toos)
        result = server(SURVEY_START_MJD + 11.0)
        self.assertEqual([too.id for too in result], ["b"])

    def test_call_boundary_is_exclusive_at_start(self):
        # Condition is (mjd > start) & (mjd < end) -> strict inequalities.
        toos = [make_too("a", mjd_start=SURVEY_START_MJD, duration=2.0)]
        server = SimTargetooServer(toos)
        self.assertIsNone(server(SURVEY_START_MJD))  # exactly at start -> excluded

    def test_call_boundary_is_exclusive_at_end(self):
        toos = [make_too("a", mjd_start=SURVEY_START_MJD, duration=2.0)]
        server = SimTargetooServer(toos)
        self.assertIsNone(server(SURVEY_START_MJD + 2.0))  # exactly at end -> excluded

    def test_call_returns_objects_not_indices(self):
        toos = [make_too("a", mjd_start=SURVEY_START_MJD, duration=2.0)]
        server = SimTargetooServer(toos)
        result = server(SURVEY_START_MJD + 1.0)
        self.assertIs(result[0], toos[0])


if __name__ == "__main__":
    unittest.main()
