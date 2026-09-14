import unittest

import healpy as hp
import numpy as np

from rubin_scheduler.site_models import CloudMap, CloudModel


class TestCloudModel(unittest.TestCase):
    def test_call_legacy_interface(self):
        """Backwards compatibility: the original placeholder
        interface, CloudModel()(cloud_value, altitude) broadcasting
        a single value over the sky, still works (with a
        FutureWarning)."""
        cloud_model = CloudModel()
        in_cloud = 1.53
        efd_data = {"cloud": in_cloud}
        alt = np.zeros(50, float)
        az = np.zeros(50, float)
        target_dict = {"altitude": alt, "azimuth": az}
        with self.assertWarns(FutureWarning):
            out_cloud = cloud_model(efd_data, target_dict)["cloud"]
        # Test that we propagated cloud value over the whole sky.
        self.assertEqual(in_cloud, out_cloud.max())
        self.assertEqual(in_cloud, out_cloud.min())
        self.assertEqual(len(out_cloud), len(alt))

    def test_call_not_enough_history(self):
        """With 0 or 1 frames, __call__ falls back to the most
        recently observed frame (or zeros if there is none)."""
        nside = 32
        n_pix = hp.nside2npix(nside)
        cloud_model = CloudModel(nside_out=nside)

        # no frames at all; falls back to zeros
        empty_history = {"mjd": np.array([]), "extinction": []}
        out = cloud_model(empty_history, 1.0)["extinction"]
        assert np.size(out) == n_pix
        assert np.all(out == 0)

        # exactly one frame; not enough for a velocity fit, falls
        # back to that frame unmodified
        rng = np.random.default_rng(seed=42)
        one_frame = rng.random(n_pix)
        history = {"mjd": np.array([0.0]), "extinction": [one_frame]}
        out = cloud_model(history, 1 / 1440.0)["extinction"]
        np.testing.assert_array_equal(out, one_frame)

    def test_call_dict_interface(self):
        """__call__ accepts a plain dict with 'mjd'/'extinction'
        keys (e.g. assembled by the caller from an EFD query)."""
        nside = 32
        n_pix = hp.nside2npix(nside)
        cloud_model = CloudModel(nside_out=nside)
        rng = np.random.default_rng(seed=1)

        mjds = np.arange(5, dtype=float)
        maps = [rng.random(n_pix) for _ in mjds]
        history = {"mjd": mjds, "extinction": maps}

        out = cloud_model(history, mjds[-1] + 0.5 / 1440.0)["extinction"]
        assert np.size(out) == n_pix
        assert np.all(np.isfinite(out))

    def test_call_cloudmap_interface(self):
        """__call__ also accepts a CloudMap-like object directly
        (anything with .mjds / .cloud_extinction_hparrays), matching
        the SeeingModel/CloudMap calling convention."""
        nside = 32
        n_pix = hp.nside2npix(nside)
        cloud_model = CloudModel(nside_out=nside)
        cm = CloudMap(nside_out=nside)
        rng = np.random.default_rng(seed=2)

        for mjd in np.arange(5, dtype=float):
            cm.add_frame(rng.random(n_pix), mjd)

        out = cloud_model(cm, cm.mjds[-1] + 0.5 / 1440.0)["extinction"]
        assert np.size(out) == n_pix

    def test_call_too_far_future_falls_back(self):
        """Requesting an mjd beyond max_extrapolate_min falls back
        to the most recently observed frame, unmodified."""
        nside = 32
        n_pix = hp.nside2npix(nside)
        cloud_model = CloudModel(nside_out=nside, max_extrapolate_min=10.0)
        rng = np.random.default_rng(seed=3)

        mjds = np.arange(5, dtype=float)
        maps = [rng.random(n_pix) for _ in mjds]
        history = {"mjd": mjds, "extinction": maps}

        # 20 minutes ahead > max_extrapolate_min=10 -> fallback
        out = cloud_model(history, mjds[-1] + 20 / 1440.0)["extinction"]
        np.testing.assert_array_equal(out, maps[-1])


if __name__ == "__main__":
    unittest.main()
