import unittest

import numpy as np

from rubin_scheduler.site_models import SeeingModel


class TestSeeingModel(unittest.TestCase):
    def test_fwhm_system_zenith(self):
        # Check calculation is being done as expected.
        seeing_model = SeeingModel()
        self.assertAlmostEqual(seeing_model.fwhm_system_zenith, 0.39862262855989494, places=7)

    def test_fwhm_geom_eff(self):
        # Check that the translation between FWHM effective and
        # geometric is done as expected.
        # (note that fwhmEff_tofwhmGeom & fwhmGeom_to_fwhmEff are
        # static methods) Document-20160 for reference.
        fwhm_eff = 1.23
        fwhm_geom = 0.822 * fwhm_eff + 0.052
        self.assertEqual(fwhm_geom, SeeingModel.fwhm_eff_to_fwhm_geom(fwhm_eff))
        self.assertEqual(fwhm_eff, SeeingModel.fwhm_geom_to_fwhm_eff(fwhm_geom))

    def test_call(self):
        # Check the calculation from fwhm_500 to fwhm_eff/fwhm_geom.
        # Use simple effective wavelengths and airmass values.
        band_list = ["500", "1000"]
        effwavelens = np.array([500.0, 1000.0])
        seeing_model = SeeingModel(band_list=band_list, eff_wavelens=effwavelens)
        # Simple fwhm_500 input.
        fwhm_500 = 1.0
        # Single airmass.
        airmass = 1.0
        seeing = seeing_model(fwhm_500, airmass)
        fwhm_eff = seeing["fwhmEff"]
        # Check shape of returned values.
        self.assertEqual(fwhm_eff.shape, (len(seeing_model.eff_wavelens),))
        # Check actual value of seeing in @ wavelen[0] @ zenith
        # after addition of system.
        fwhm_system = seeing_model.fwhm_system_zenith
        expected_fwhm_eff = 1.16 * np.sqrt(fwhm_system**2 + 1.04 * fwhm_500**2)
        self.assertAlmostEqual(fwhm_eff[0], expected_fwhm_eff, 15)
        # Check expected value if we remove the system component.
        seeing_model.fwhm_system_zenith = 0
        seeing = seeing_model(fwhm_500, airmass)
        expected_fwhm_eff = 1.16 * np.sqrt(1.04) * fwhm_500
        self.assertAlmostEqual(seeing["fwhmEff"][0], expected_fwhm_eff, 15)
        # Check scaling with wavelength (remove system component).
        expected_fwhm_eff = 1.16 * np.sqrt(1.04) * fwhm_500 * np.power(500.0 / effwavelens[1], 0.3)
        self.assertAlmostEqual(seeing["fwhmEff"][1], expected_fwhm_eff, places=15)
        # Multiple airmasses.
        airmass = np.array([1.0, 1.5])
        seeing = seeing_model(fwhm_500, airmass)
        self.assertEqual(seeing["fwhmEff"].shape, (len(seeing_model.eff_wavelens), len(airmass)))
        expected_fwhm_eff = fwhm_500 * 1.16 * np.sqrt(1.04)
        self.assertEqual(seeing["fwhmEff"][0][0], expected_fwhm_eff)
        # Check scaling with airmass.
        expected_fwhm_eff = expected_fwhm_eff * np.power(airmass[1], 0.6)
        self.assertAlmostEqual(seeing["fwhmEff"][0][1], expected_fwhm_eff, places=15)

    def test_wind_seeing(self):
        # The wind + dome-temperature term, combined in quadrature.
        seeing_model = SeeingModel()
        fwhm_500 = 0.7
        airmass = 1.2
        base = seeing_model(fwhm_500, airmass)["fwhmEff"]
        # No wind info -> identical to the historical call.
        self.assertTrue(np.array_equal(base, seeing_model(fwhm_500, airmass, delta_t=2.0)["fwhmEff"]))
        params = seeing_model.wind_seeing_params
        # Calm limit: uniform d0 + d1 * deltaT^2 added in quadrature, any azimuth.
        calm = seeing_model(fwhm_500, airmass, wind_speed=0.0, wind_direction=0.0, azimuth=1.0, delta_t=1.5)
        expected = np.sqrt(base**2 + params["d0"] + params["d1"] * 1.5**2)
        self.assertTrue(np.allclose(calm["fwhmEff"], expected))
        # Pointing into a moderate wind flushes the dome: the added seeing is
        # a small fraction of the calm value.
        upwind = seeing_model(fwhm_500, airmass, wind_speed=8.0, wind_direction=0.0, azimuth=0.0, delta_t=1.5)
        added_calm = calm["fwhmEff"] ** 2 - base**2
        added_upwind = upwind["fwhmEff"] ** 2 - base**2
        self.assertTrue(np.all(added_upwind < 0.1 * added_calm))
        # Downwind the warm dome is never flushed: less seeing upwind than downwind.
        downwind = seeing_model(fwhm_500, airmass, wind_speed=8.0, wind_direction=0.0, azimuth=np.pi, delta_t=1.5)
        self.assertTrue(np.all(upwind["fwhmEff"] < downwind["fwhmEff"]))
        # Downwind (cos theta = -1) the exponential is exactly 1 at any speed:
        # unflushed dome term d0 + d1 * deltaT^2, plus the wake (t + s) (2 v)^2.
        expected = np.sqrt(base**2 + params["d0"] + params["d1"] * 1.5**2
                           + (params["t"] + params["s"]) * (2 * 8.0) ** 2)
        self.assertTrue(np.allclose(downwind["fwhmEff"], expected))
        # Only a warm dome adds seeing; a cold dome is harmless.
        warm = seeing_model(fwhm_500, airmass, wind_speed=0.5, wind_direction=0.0, azimuth=0.0, delta_t=2.0)
        cold = seeing_model(fwhm_500, airmass, wind_speed=0.5, wind_direction=0.0, azimuth=0.0, delta_t=-2.0)
        no_dt = seeing_model(fwhm_500, airmass, wind_speed=0.5, wind_direction=0.0, azimuth=0.0)
        self.assertTrue(np.all(warm["fwhmEff"] > no_dt["fwhmEff"]))
        self.assertTrue(np.allclose(cold["fwhmEff"], no_dt["fwhmEff"]))
        # Array airmass with per-pointing azimuths broadcasts over bands.
        airmasses = np.array([1.0, 1.2, 1.5])
        azimuths = np.radians(np.array([0.0, 90.0, 180.0]))
        seeing = seeing_model(fwhm_500, airmasses, wind_speed=8.0, wind_direction=0.0, azimuth=azimuths)
        self.assertEqual(seeing["fwhmEff"].shape, (len(seeing_model.eff_wavelens), len(airmasses)))
        # Custom parameters propagate.
        custom = SeeingModel(wind_seeing_params=dict(d0=0.0, d1=0.0, v0=5.0, t=1e-3, s=1e-3))
        wake = custom(fwhm_500, airmass, wind_speed=10.0, wind_direction=0.0, azimuth=np.pi)
        expected = np.sqrt(custom(fwhm_500, airmass)["fwhmEff"] ** 2 + 2e-3 * (10.0 * 2) ** 2)
        self.assertTrue(np.allclose(wake["fwhmEff"], expected))

    def test_airmass_scale_system(self):
        # With airmass_scale_system=False the hardware floor stays at its
        # zenith value; the atmosphere still scales with airmass.
        fwhm_500 = 0.7
        default_model = SeeingModel()
        flat_model = SeeingModel(airmass_scale_system=False)
        # Identical at zenith.
        self.assertTrue(
            np.allclose(default_model(fwhm_500, 1.0)["fwhmEff"], flat_model(fwhm_500, 1.0)["fwhmEff"])
        )
        # Smaller prediction at airmass > 1.
        self.assertTrue(
            np.all(flat_model(fwhm_500, 1.5)["fwhmEff"] < default_model(fwhm_500, 1.5)["fwhmEff"])
        )
        # Exact value: floor held at zenith.
        airmass = 1.5
        atmo = fwhm_500 * (500.0 / flat_model.eff_wavelens) ** 0.3 * airmass**0.6
        expected = 1.16 * np.sqrt(flat_model.fwhm_system_zenith**2 + 1.04 * atmo**2)
        self.assertTrue(np.allclose(flat_model(fwhm_500, airmass)["fwhmEff"], expected))


if __name__ == "__main__":
    unittest.main()
