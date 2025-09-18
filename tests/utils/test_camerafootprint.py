import unittest

import numpy as np

from rubin_scheduler.utils import LsstCameraFootprint


class TestLsstcamerafootprint(unittest.TestCase):
    def setUp(self):
        self.obj_ra = np.array([10.0, 12.1], float)
        self.obj_dec = np.array([-30.0, -30.0], float)
        self.obs_ra = np.array([10.0, 10.0], float)
        self.obs_dec = np.array([-30.0, -30.0], float)
        self.obs_rot_sky_pos = np.zeros(2)

    def test_camera(self):
        camera = LsstCameraFootprint(
            units="degrees",
        )
        idx_obs = camera(self.obj_ra, self.obj_dec, self.obs_ra, self.obs_dec, self.obs_rot_sky_pos)
        # The first of these objects should be in the middle of
        # the FOV, while the second is outside
        self.assertEqual(idx_obs, [0])

        # Check that lower max radius means
        # fewer valid pixels
        c1 = LsstCameraFootprint(max_radius=1.75)
        c2 = LsstCameraFootprint(max_radius=1.94)

        assert c1.camera_fov.sum() < c2.camera_fov.sum()


if __name__ == "__main__":
    unittest.main()
