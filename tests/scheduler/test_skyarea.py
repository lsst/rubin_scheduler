import os
import unittest

import healpy as hp
import numpy as np

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.utils import (
    EuclidOverlapFootprint,
    PerFilterStep,
    SkyAreaGenerator,
    SkyAreaGeneratorGalplane,
    generate_all_sky,
    make_rolling_footprints,
)

datadir = os.path.join(get_data_dir(), "scheduler")


class TestSkyArea(unittest.TestCase):
    def setUp(self):
        self.nside = 32

    def test_perfilterstep(self):
        survey_length = 13
        pfs = PerFilterStep(
            loaded_dict={"u": np.array([0, 1, 2, 3, 10, 11])},
            survey_length=survey_length,
        )
        # check at day 1
        t1 = pfs(1, np.array([0]))

        # check at day 7
        t7 = pfs(7, np.array([0]))

        t_end = pfs(survey_length, np.array([0]))

        # u should grow fast at the start
        assert t1[0] > t1[2]

        # i and r should always grow the same
        assert t1[2] == t1[3]
        assert t7[2] == t7[3]

        # Everthing should be one at the end
        assert np.unique(t_end) == 1

    def test_make_rolling(self):
        """Check that we can make rolling footprints"""
        # Check that we can run with no kwargs
        footprints = make_rolling_footprints()

        assert footprints is not None

        # check that various kwarg combos work
        for nslice in [2, 3]:
            for uniform in [True, False]:
                for order_roll in [0, 1]:
                    for n_cycles in [3, 4]:
                        footprints = make_rolling_footprints(
                            nslice=nslice,
                            scale=0.8,
                            nside=32,
                            wfd_indx=None,
                            order_roll=order_roll,
                            n_cycles=n_cycles,
                            n_constant_start=2,
                            n_constant_end=6,
                            verbose=False,
                            uniform=uniform,
                        )

                        assert footprints is not None

    @unittest.skipUnless(os.path.isdir(datadir), "Test data not available.")
    def test_skyareagenerator(self):
        # Just test that it sets up and returns maps
        s = SkyAreaGenerator(nside=self.nside)
        footprints, labels = s.return_maps()
        expected_labels = ["", "LMC_SMC", "bulge", "dusty_plane", "lowdust", "nes", "scp", "virgo"]
        self.assertEqual(set(np.unique(labels)), set(expected_labels))
        # Check that ratios in the low-dust wfd in r band are 1
        # This doesn't always have to be the case, but should be with defaults
        lowdust = np.where(labels == "lowdust")
        self.assertTrue(np.all(footprints["r"][lowdust] == 1))

    @unittest.skipUnless(os.path.isdir(datadir), "Test data not available.")
    def test_skyareagenerator_nside(self):
        # Just check two other likely common nsides
        for nside in (16, 64):
            s = SkyAreaGenerator(nside=nside)
            footprints, labels = s.return_maps()
            lowdust = np.where(labels == "lowdust")
            self.assertTrue(np.all(footprints["r"][lowdust] == 1))

    @unittest.skipUnless(os.path.isdir(datadir), "Test data not available.")
    def test_skyareageneratorgalplane(self):
        # Just test that it sets up and returns maps
        s = SkyAreaGeneratorGalplane(nside=self.nside)
        footprints, labels = s.return_maps()
        expected_labels = ["", "LMC_SMC", "bulgy", "dusty_plane", "lowdust", "nes", "scp", "virgo"]
        self.assertEqual(set(np.unique(labels)), set(expected_labels))
        # Check that ratios in the low-dust wfd in r band are 1
        # This doesn't always have to be the case, but should be with defaults
        lowdust = np.where(labels == "lowdust")
        self.assertTrue(np.all(footprints["r"][lowdust] == 1))

    @unittest.skipUnless(os.path.isdir(datadir), "Test data not available.")
    def test_euclidoverlapfootprint(self):
        # Just test that it sets up and returns maps
        s = EuclidOverlapFootprint(nside=self.nside)
        footprints, labels = s.return_maps()
        expected_labels = [
            "",
            "LMC_SMC",
            "bulgy",
            "dusty_plane",
            "lowdust",
            "nes",
            "scp",
            "virgo",
            "euclid_overlap",
        ]
        self.assertEqual(set(np.unique(labels)), set(expected_labels))
        # Check that ratios in the low-dust wfd in r band are 1
        # This doesn't always have to be the case, but should be with defaults
        lowdust = np.where(labels == "lowdust")
        self.assertTrue(np.all(footprints["r"][lowdust] == 1))

    def test_generate_all_sky(self):
        # Test that the utility generate_all_sky returns appropriately
        nside = 32
        sky = generate_all_sky(nside=nside)
        expected_keys = ["map", "ra", "dec", "eclip_lat", "eclip_lon", "gal_lat", "gal_lon"]
        for k in expected_keys:
            self.assertTrue(k in sky.keys())
        self.assertEqual(sky["ra"].size, hp.nside2npix(nside))


if __name__ == "__main__":
    unittest.main()
