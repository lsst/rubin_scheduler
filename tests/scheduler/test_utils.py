import os
import unittest

import healpy as hp
import numpy as np

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import example_scheduler, run_sched
from rubin_scheduler.scheduler.model_observatory import KinemModel, ModelObservatory
from rubin_scheduler.scheduler.utils import (
    SchemaConverter,
    empty_observation,
    make_rolling_footprints,
    restore_scheduler,
    run_info_table,
    season_calc,
)
from rubin_scheduler.utils import survey_start_mjd


class TestUtils(unittest.TestCase):
    def tearDownClass():
        os.remove("temp.sqlite")

    @unittest.skipUnless(
        os.path.isfile(os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")),
        "Test data not available.",
    )
    def test_nside(self):
        """Test the example scheduler can be set to different nsides."""
        mjd_start = survey_start_mjd()
        _ = example_scheduler(mjd_start=mjd_start, nside=64)
        _ = example_scheduler(mjd_start=mjd_start, nside=8)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")),
        "Test data not available.",
    )
    def test_example(self):
        """Test the example scheduler executes all the expected surveys"""
        mjd_start = survey_start_mjd()
        scheduler = example_scheduler(mjd_start=mjd_start)
        observatory, scheduler, observations = run_sched(scheduler, mjd_start=mjd_start, survey_length=7)
        u_notes = np.unique(observations["note"])

        # Note that some of these may change and need to be updated if
        # survey start date changes, e.g., different DDFs in season,
        # or different lunar phase means different filters get picked
        # for the blobs
        notes_to_check = [
            "blob_long, gr, a",
            "blob_long, gr, b",
            "greedy",
            "long",
            "pair_15, iz, a",
            "pair_15, iz, b",
            "pair_15, ri, a",
            "pair_15, ri, b",
            "pair_15, yy, a",
            "pair_15, yy, b",
            "pair_15, zy, a",
            "pair_33, gr, a",
            "pair_33, gr, b",
            "pair_33, ri, a",
            "pair_33, ug, a",
            "pair_33, ug, b",
            "pair_33, yy, a",
            "pair_33, yy, b",
            "twilight_near_sun, 0",
            "twilight_near_sun, 1",
            "twilight_near_sun, 2",
            "twilight_near_sun, 3",
        ]

        for note in notes_to_check:
            assert note in u_notes

        for note in u_notes:
            # If this fails, time to add something to notes_to_check
            assert note in u_notes

    @unittest.skipUnless(
        os.path.isfile(os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")),
        "Test data not available.",
    )
    def test_start_of_night_example(self):
        """Test the example scheduler starting at the beginnig of a night."""
        mjd_start = survey_start_mjd()
        scheduler = example_scheduler(mjd_start=mjd_start)
        observatory, scheduler, observations = run_sched(
            scheduler, mjd_start=mjd_start - 0.5, survey_length=5
        )

    @unittest.skipUnless(
        os.path.isfile(os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")),
        "Test data not available.",
    )
    def test_altaz_limit(self):
        """Test that setting some azimuth limits via the kinematic
        model works"""
        mjd_start = survey_start_mjd()
        scheduler = example_scheduler(mjd_start=mjd_start)
        km = KinemModel(mjd0=mjd_start)
        km.setup_telescope(az_limits=[[0.0, 90.0], [270.0, 360.0]])
        mo = ModelObservatory(mjd_start=mjd_start, kinem_model=km)

        mo, scheduler, observations = sim_runner(
            mo,
            scheduler,
            survey_length=3.0,
            verbose=False,
            filename=None,
        )

        az = np.degrees(observations["az"])
        forbidden = np.where((az > 90) & (az < 270))[0]
        # Let a few pairs try to complete since by default we don't
        # use an agressive shadow_minutes
        n_forbidden = np.size(
            [obs for obs in observations[forbidden]["note"] if (("pair_33" not in obs) | (", b" not in obs))]
        )

        assert n_forbidden == 0

        km = KinemModel(mjd0=mjd_start)
        km.setup_telescope(alt_limits=[[40.0, 70.0]])
        mo = ModelObservatory(mjd_start=mjd_start, kinem_model=km)

        mo, scheduler, observations = sim_runner(
            mo,
            scheduler,
            survey_length=3.0,
            verbose=False,
            filename=None,
        )
        alt = np.degrees(observations["alt"])
        n_forbidden = np.size(np.where((alt > 70) & (alt < 40))[0])

        assert n_forbidden == 0

    def test_restore(self):
        """Test we can restore a scheduler properly"""
        # MJD set so it's in test data range
        mjd_start = survey_start_mjd()
        n_visit_limit = 3000

        scheduler = example_scheduler(mjd_start=mjd_start)

        mo = ModelObservatory(mjd_start=mjd_start)
        # Never load too many nights of sky
        mo.sky_model.load_length = 10.0
        mo, scheduler, observations = sim_runner(
            mo,
            scheduler,
            survey_length=30.0,
            verbose=False,
            filename=None,
            n_visit_limit=n_visit_limit,
        )

        # Won't be exact if we restart in the middle of a blob sequence
        # since the queue isn't reconstructed.
        # Also, any scripted observations that get generated
        # during the night (e.g., long gaps observations) will get lost,
        # so need to restart on a new night to ensure identical results.

        nd = np.zeros(observations.size)
        nd[1:] = np.diff(observations["night"])

        break_indx = np.min(np.where((observations["ID"] >= n_visit_limit / 2.0) & (nd != 0))[0])
        new_n_limit = n_visit_limit - break_indx

        new_mo = ModelObservatory(mjd_start=mjd_start)
        # Never load too much sky
        new_mo.sky_model.load_length = 10.0
        new_sched = example_scheduler(mjd_start=mjd_start)

        # Restore some of the observations
        new_sched, new_mo = restore_scheduler(break_indx - 1, new_sched, new_mo, observations, fast=False)

        # Simulate ahead and confirm that it behaves the same
        # as running straight through
        new_mo, new_sched, new_obs = sim_runner(
            new_mo,
            new_sched,
            survey_length=20.0,
            verbose=False,
            filename=None,
            n_visit_limit=new_n_limit,
        )

        # Check that observations taken after restart match those from
        # before Jenkins can be bad at comparing things, so if it thinks
        # they aren't the same, check column-by-column to double check
        if not np.all(new_obs == observations[break_indx:]):
            names = new_obs.dtype.names
            for name in names:
                # If it's a string
                if new_obs[name].dtype == "<U40":
                    assert np.all(new_obs[name] == observations[break_indx:][name])
                # Otherwise should be number-like
                else:
                    assert np.allclose(new_obs[name], observations[break_indx:][name])
        # Didn't need to go by column, the observations after restart
        # match the ones that were taken all at once.
        else:
            assert np.all(new_obs == observations[break_indx:])

        # And again, but this time using the fast array restore
        new_mo = ModelObservatory(mjd_start=mjd_start)
        new_mo.sky_model.load_length = 10.0
        new_sched = example_scheduler(mjd_start=mjd_start)
        new_sched, new_mo = restore_scheduler(break_indx - 1, new_sched, new_mo, observations, fast=True)
        # Simulate ahead and confirm that it behaves the same as
        # running straight through
        new_mo, new_sched, new_obs_fast = sim_runner(
            new_mo,
            new_sched,
            survey_length=20.0,
            verbose=False,
            filename=None,
            n_visit_limit=new_n_limit,
        )

        # Check that observations taken after restart match those
        # from before Jenkins can be bad at comparing things, so if
        # it thinks they aren't the same, check column-by-column to
        # double check
        if not np.all(new_obs_fast == observations[break_indx:]):
            names = new_obs_fast.dtype.names
            for name in names:
                # If it's a string
                if new_obs_fast[name].dtype == "<U40":
                    assert np.all(new_obs_fast[name] == observations[break_indx:][name])
                # Otherwise should be number-like
                else:
                    assert np.allclose(new_obs_fast[name], observations[break_indx:][name])
        # Didn't need to go by column, the observations after restart
        # match the ones that were taken all at once.
        else:
            assert np.all(new_obs_fast == observations[break_indx:])

    def test_season(self):
        """
        Test that the season utils work as intended
        """
        night = 365.25 * 3.5
        plain = season_calc(night)
        assert plain == 3

        mod2 = season_calc(night, modulo=2)
        assert mod2 == 1

        mod3 = season_calc(night, modulo=3)
        assert mod3 == 0

        mod3 = season_calc(night, modulo=3, max_season=2)
        assert mod3 == -1

        mod3 = season_calc(night, modulo=3, max_season=2, offset=-365.25 * 2)
        assert mod3 == 1

        mod3 = season_calc(night, modulo=3, max_season=2, offset=-365.25 * 10)
        assert mod3 == -1

        mod3 = season_calc(night, modulo=3, offset=-365.25 * 10)
        assert mod3 == -1

    def test_schema_convert(self):
        sc = SchemaConverter()

        n = 20
        obs = empty_observation(n=n)
        obs["ID"] = np.arange(n)

        # check that we can write observations to a database
        sc.obs2opsim(obs, filename="temp.sqlite")

        # change the ID and write it again, should append
        obs["ID"] = np.arange(n) + n + 1
        sc.obs2opsim(obs, filename="temp.sqlite")

        read_in = sc.opsim2obs("temp.sqlite")

        assert len(read_in) == 2 * n

    def test_run_info_table(self):
        """Test run_info_table gets information"""
        observatory = ModelObservatory(nside=8, mjd_start=survey_start_mjd())
        version_info = run_info_table(observatory)
        # Make a minimal set of keys that ought to be in the info table
        # Update these if the value they're stored as changes
        # (either from run_info_table or observatory.info)
        need_keys = [
            "rubin_scheduler.__version__",
            "hostname",
            "Date, ymd",
            "site_models",
            "skybrightness_pre",
        ]
        have_keys = list(version_info["Parameter"])
        for k in need_keys:
            self.assertTrue(k in have_keys)

    def test_make_rolling_footprints_uniform(self):
        """Test that we can make a uniform folling footprint"""

        # utility function to get sun ra at a given mjd
        from astropy.coordinates import EarthLocation, get_sun
        from astropy.time import Time

        def _get_sun_ra_at_mjd(mjd):
            t = Time(
                mjd,
                format="mjd",
                location=EarthLocation.of_site("Cerro Pachon"),
            )
            return get_sun(t).ra.deg

        nside = 32
        scale = 0.9
        order_roll = 1

        # setup a simple test survey with dec < 20
        _, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
        base_footprint = np.zeros(hp.nside2npix(nside), dtype=float)
        wfd_indx = np.where(dec < 20)[0]
        base_footprint[wfd_indx] = 1

        # now test uniformity of uniform rolling and baseline at a range
        # of survey start times
        for mjd_offset in [0, 12, 60, 90, 180, 195, 240, 267, 270, 319]:
            mjd_start = survey_start_mjd() + mjd_offset
            sun_ra_start = _get_sun_ra_at_mjd(mjd_start)
            fp_hp = {
                "g": base_footprint.copy(),
                "r": base_footprint.copy(),
                "i": base_footprint.copy(),
            }

            for nslice in [2, 3]:
                uniform_footprint = make_rolling_footprints(
                    fp_hp=fp_hp,
                    mjd_start=mjd_start,
                    sun_ra_start=sun_ra_start,
                    nslice=nslice,
                    scale=scale,
                    nside=nside,
                    wfd_indx=wfd_indx,
                    order_roll=order_roll,
                    n_cycles=None,
                    n_constant_start=None,
                    n_constant_end=6,
                    uniform=True,
                )

                footprint = make_rolling_footprints(
                    fp_hp=fp_hp,
                    mjd_start=mjd_start,
                    sun_ra_start=sun_ra_start,
                    nslice=nslice,
                    scale=scale,
                    nside=nside,
                    wfd_indx=wfd_indx,
                    order_roll=order_roll,
                    n_cycles=None,
                    n_constant_start=None,
                    n_constant_end=6,
                    uniform=False,
                )

                # make sure all of the bands are OK
                for band in ["g", "r", "i"]:
                    # look at each yearly release
                    for i in range(1, 11):
                        mjd = mjd_start + i * 365.25
                        ud = uniform_footprint(mjd, norm=False)
                        uniform_scat = np.std(ud[band][wfd_indx])

                        # uniform rolling surveys have special times where the
                        # scatter in the requested number of observations is 0
                        if nslice == 2:
                            if i in [1, 4, 7, 10]:
                                self.assertTrue(np.allclose(uniform_scat, 0))
                            else:
                                self.assertTrue(uniform_scat != 0)
                        elif nslice == 3:
                            if i in [1, 5, 9, 10]:
                                self.assertTrue(np.allclose(uniform_scat, 0))
                            else:
                                self.assertTrue(uniform_scat != 0)

                        # non-uniform rolling surveys only have uniform
                        # requests at Y1 and Y10
                        d = footprint(mjd, norm=False)
                        scat = np.std(d[band][wfd_indx])

                        if i in [1, 10]:
                            self.assertTrue(np.allclose(scat, 0))
                        else:
                            self.assertTrue(scat != 0)


if __name__ == "__main__":
    unittest.main()
