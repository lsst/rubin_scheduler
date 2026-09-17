__all__ = ("SeeingModel", "WIND_SEEING_DEFAULTS")

import warnings

import numpy as np

from rubin_scheduler.utils import SysEngVals

# Default parameters for the wind + dome-temperature seeing term:
#   FWHM_wind^2 = [d0 + d1 * max(deltaT, 0)^2] *
#                 exp(-v * (1 + cos(theta)) / 2 / v0)
#                 + (t + s) * (v * (1 - cos(theta)))^2      [arcsec^2]
# where v is the wind speed (m/s), deltaT the dome-minus-outdoor temperature
# difference (K), and theta the angle between the pointing azimuth and the
# direction the wind comes from (theta=0 -> pointing into the wind, which
# flushes the dome). The term combines with the atmospheric + system FWHM in
# quadrature. Values are the "nightly-baseline" fit to LSSTCam ConsDB visits
# (2025-10 .. 2026-06, science programs BLOCK-365/407/408/419/421, ~28k
# visits with EFD dome/outdoor temperatures): each night's median measured
# zenith/500nm-corrected seeing was profiled out, so the parameters are
# identified purely from within-night contrasts in pointing-vs-wind angle,
# wind speed and deltaT. Note the fit found no (v*(1-cos theta))^2 wake
# signal at the wind speeds sampled (<~15 m/s), so t = s = 0; a cold dome
# (deltaT < 0) adds no seeing.
WIND_SEEING_DEFAULTS = dict(d0=0.341, d1=0.112, v0=5.85, t=0.0, s=0.0)


class SeeingModel:
    """LSST FWHM calculations for FWHM_effective and FWHM_geometric.

    Calculations of the delivered values are based on equations in
    Document-20160 ("Atmospheric and Delivered Image Quality in OpSim"
    by Bo Xin, George Angeli, Zeljko Ivezic)
    An example of the calculation of delivered image seeing from DIMM
    FWHM_500 is available in
    https://smtn-002.lsst.io/
    #calculating-m5-values-in-the-lsst-operations-simulator

    Parameters
    ----------
    band_list : `list` [`str`], opt
        List of the band bandpasses for which to calculate delivered
        FWHM_effective and FWHM_geometric
        Default ['u', 'g', 'r', 'i', 'z', 'y']
    eff_wavelens : `list` [`float`] or None, opt
        Effective wavelengths for those bandpasses, in nanometers.
        If None, loades from rubin_scheduler.utils.SysEngVals
    telescope_seeing : `float`, opt
        The contribution to the delivered FWHM from the telescope,
        in arcseconds. Default 0.25"
    optical_design_seeing : `float`, opt
        The contribution to the seeing from the optical design, in
        arcseconds. Default 0.08 arcseconds
    camera_seeing : `float`, opt
        The contribution to the seeing from the camera, in arcseconds.
        Default 0.30 arcseconds
    raw_seeing_wavelength : `float`, opt
        The wavelength of the DIMM-delivered equivalent FWHM, in
        nanometers. Default 500nm.
    efd_seeing : `str`, opt
        The name of the DIMM FWHM measurements in the efd /
        conditions object. Default `FWHM_500`
    wind_seeing_params : `dict` or None, opt
        Parameters (d0, d1, v0, t, s) for the wind + dome-temperature
        seeing term (see WIND_SEEING_DEFAULTS for the definition and
        provenance). Default None uses WIND_SEEING_DEFAULTS. The term is
        only applied when wind information is passed to __call__.
    airmass_scale_system : `bool`, opt
        If True (default, the historical behavior), the system
        contribution to the FWHM scales with airmass^0.6 like the
        atmosphere. If False, the system contribution is held at its
        zenith value (the hardware does not know about airmass); fits to
        LSSTCam ConsDB visits prefer this variant.
    filter_list : `list` [`str`], opt
        Deprecated version of band_list
    """

    def __init__(
        self,
        band_list=["u", "g", "r", "i", "z", "y"],
        eff_wavelens=None,
        telescope_seeing=0.25,
        optical_design_seeing=0.08,
        camera_seeing=0.30,
        raw_seeing_wavelength=500,
        efd_seeing="FWHM_500",
        wind_seeing_params=None,
        airmass_scale_system=True,
        filter_list=None,
    ):
        if filter_list is not None:
            warnings.warn("filter_list deprecated in favor of band_list", FutureWarning)
            band_list = filter_list
        self.band_list = band_list
        # Setting self.filter_list for backward compatibility with ts_scheduler
        self.filter_list = band_list
        if eff_wavelens is None:
            sev = SysEngVals()
            eff_wavelens = [sev.eff_wavelengths[f] for f in band_list]
        self.eff_wavelens = np.array(eff_wavelens)
        self.telescope_seeing = telescope_seeing
        self.optical_design_seeing = optical_design_seeing
        self.raw_seeing_wavelength = raw_seeing_wavelength
        self.efd_seeing = efd_seeing
        self.camera_seeing = camera_seeing
        self.wind_seeing_params = (
            dict(WIND_SEEING_DEFAULTS) if wind_seeing_params is None else dict(wind_seeing_params)
        )
        self.airmass_scale_system = airmass_scale_system

        self._set_fwhm_zenith_system()

    def configure(self):
        """Deprecated. Configure through the init method."""
        warnings.warn("the configure method is deprecated")

    def config_info(self):
        """Deprecated. Report configuration parameters and version
        information."""
        warnings.warn("the config_info method is deprecated.")

    def _set_fwhm_zenith_system(self):
        """Calculate the system contribution to FWHM at zenith.

        This is simply the individual telescope, optics, and camera
        contributions combined in quadrature.
        """
        self.fwhm_system_zenith = np.sqrt(
            self.telescope_seeing**2 + self.optical_design_seeing**2 + self.camera_seeing**2
        )

    def wind_seeing_squared(self, wind_speed, wind_direction, azimuth, delta_t=0.0):
        """The squared wind + dome-temperature seeing term (arcsec^2).

        FWHM_wind^2 = [d0 + d1 * max(deltaT, 0)^2]
                          * exp(-v * (1 + cos(theta)) / 2 / v0)
                      + (t + s) * (v * (1 - cos(theta)))^2

        with theta = azimuth - wind_direction. Combines with the delivered
        FWHM in quadrature: fwhm_eff = sqrt(fwhm_eff^2 + FWHM_wind^2).
        Pointing into the wind (theta = 0) flushes the dome; only a dome
        warmer than the outside air (deltaT > 0) adds dome seeing.
        Parameters default to WIND_SEEING_DEFAULTS (the nightly-baseline fit
        to ConsDB visits); override via the wind_seeing_params init argument.

        Parameters
        ----------
        wind_speed : `float` or `np.ndarray`
            Wind speed (m/s).
        wind_direction : `float` or `np.ndarray`
            Direction the wind originates from, in radians
            (0 = from the north, pi/2 = from the east - the same
            convention as Conditions.wind_direction).
        azimuth : `float` or `np.ndarray`
            Pointing azimuth, in radians.
        delta_t : `float` or `np.ndarray`, opt
            Dome-minus-outdoor temperature difference (K). Default 0.

        Returns
        -------
        fwhm_wind_squared : `float` or `np.ndarray`
            Squared seeing contribution (arcsec^2), broadcast over the
            inputs.
        """
        params = self.wind_seeing_params
        cos_theta = np.cos(np.asarray(azimuth, dtype=float) - wind_direction)
        warm = np.clip(delta_t, 0, None) ** 2
        dome = (params["d0"] + params["d1"] * warm) * np.exp(-wind_speed * (1 + cos_theta) / 2 / params["v0"])
        wake = (params["t"] + params["s"]) * (wind_speed * (1 - cos_theta)) ** 2
        return dome + wake

    def __call__(self, fwhm_z, airmass, wind_speed=None, wind_direction=None, azimuth=None, delta_t=0.0):
        """Calculate the seeing values FWHM_eff and FWHM_geom at the
        given airmasses, for the specified effective wavelengths, given
        FWHM_zenith (typically FWHM_500).

        FWHM_geom represents the geometric size of the PSF; FWHM_eff
        represents the FWHM of a single gaussian which encloses the
        same number of pixels as N_eff (the number of pixels enclosed
        in the actual PSF -- this is the value to use when calculating
        SNR).

        FWHM_geom(") = 0.822 * FWHM_eff(") + 0.052"

        The FWHM_eff includes a contribution from the system and from
        the atmosphere.  Both of these are expected to scale with
        airmass^0.6 and with (500(nm)/wavelength(nm))^0.3.
        FWHM_eff = 1.16 * sqrt(FWHM_sys**2 + 1.04*FWHM_atm**2)

        If wind information is supplied (wind_speed, wind_direction and
        azimuth all not None), the wind + dome-temperature seeing term
        (see wind_seeing_squared) is combined in quadrature with the
        atmospheric + system FWHM.

        Parameters
        ----------
        fwhm_z: `float`, or efdData `dict`
            FWHM at zenith (arcsec).
        airmass: `float`, `np.array`, or targetDict `dict`
            Airmass (unitless).
        wind_speed : `float` or `np.ndarray`, opt
            Wind speed (m/s). Default None (no wind term applied).
        wind_direction : `float` or `np.ndarray`, opt
            Direction the wind originates from, in radians (0 = from N,
            pi/2 = from E). Default None (no wind term applied).
        azimuth : `float` or `np.ndarray`, opt
            Pointing azimuth in radians, matching the airmass values.
            Default None (no wind term applied).
        delta_t : `float` or `np.ndarray`, opt
            Dome-minus-outdoor temperature difference (K). Default 0.

        Returns
        -------
        FWHMeff, FWHMGeom : `dict` of {`numpy.ndarray`, `numpy.ndarray`}
            FWHMeff, FWHMgeom: both are the same shape numpy.ndarray.
            If airmass is a single value, FWHMeff & FWHMgeom are 1-d
            arrays, with the same order as eff_wavelen (i.e.
            eff_wavelen[0] = u, then FWHMeff[0] = u). If airmass is a
            numpy array, FWHMeff and FWHMgeom are 2-d arrays, in the
            order of <band><airmass> (i.e. eff_wavelen[0] = u, 1-d
            array over airmass range).
        """
        if isinstance(fwhm_z, dict):
            fwhm_z = fwhm_z[self.efd_seeing]
        if isinstance(airmass, dict):
            airmass = airmass["airmass"]
        airmass_correction = np.power(airmass, 0.6)
        wavelen_correction = np.power(self.raw_seeing_wavelength / self.eff_wavelens, 0.3)
        # The system contribution scales with airmass only in the historical
        # configuration (airmass_scale_system=True).
        if self.airmass_scale_system:
            system_correction = airmass_correction
        else:
            system_correction = np.ones_like(np.asarray(airmass, dtype=float))
        if isinstance(airmass, np.ndarray):
            fwhm_system = self.fwhm_system_zenith * np.outer(
                np.ones(len(wavelen_correction)), system_correction
            )
            fwhm_atmo = fwhm_z * np.outer(wavelen_correction, airmass_correction)
        else:
            fwhm_system = self.fwhm_system_zenith * system_correction
            fwhm_atmo = fwhm_z * wavelen_correction * airmass_correction
        # Calculate combined FWHMeff.
        fwhm_eff = 1.16 * np.sqrt(fwhm_system**2 + 1.04 * fwhm_atmo**2)
        # Add the wind + dome-temperature seeing in quadrature, if wind
        # information was provided. The term is achromatic and broadcasts
        # over the band dimension.
        if wind_speed is not None and wind_direction is not None and azimuth is not None:
            fwhm_wind_sq = self.wind_seeing_squared(wind_speed, wind_direction, azimuth, delta_t=delta_t)
            fwhm_eff = np.sqrt(fwhm_eff**2 + fwhm_wind_sq)
        # Translate to FWHMgeom.
        fwhm_geom = self.fwhm_eff_to_fwhm_geom(fwhm_eff)
        return {"fwhmEff": fwhm_eff, "fwhmGeom": fwhm_geom}

    @staticmethod
    def fwhm_eff_to_fwhm_geom(fwhm_eff):
        """Calculate FWHM_geom from FWHM_eff.

        Parameters
        ----------
        fwhm_eff : `float` or `np.ndarray`

        Returns
        -------
        FWHM_geom : `float` or `np.ndarray`
        """
        return 0.822 * fwhm_eff + 0.052

    @staticmethod
    def fwhm_geom_to_fwhm_eff(fwhm_geom):
        """Calculate FWHM_eff from FWHM_geom.

        Parameters
        ----------
        fwhm_geom : `float` or `np.ndarray`

        Returns
        -------
        FWHM_eff : `float` or `np.ndarray`
        """
        return (fwhm_geom - 0.052) / 0.822
