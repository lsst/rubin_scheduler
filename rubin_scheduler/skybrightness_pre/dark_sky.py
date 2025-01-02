__all__ = ("dark_sky", "dark_m5")

import os

import healpy as hp
import numpy as np

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.site_models import SeeingModel
from rubin_scheduler.utils import DEFAULT_NSIDE, m5_flat_sed


def dark_sky(nside=DEFAULT_NSIDE):
    """Load an array of HEALpix maps that have the darkest expected sky
    backgrounds per band.

    Parameters
    ----------
    nside : `int`
        Desired nside resolution (default=DEFAULT_NSIDE).

    Returns
    -------
    dark_sky_data : `np.ndarray`
        Named array with dark sky data for each band.
    """
    if not hasattr(dark_sky, "data"):
        # Load up the data
        data_dir = get_data_dir()
        data = np.load(os.path.join(data_dir, "skybrightness_pre", "dark_maps.npz"))
        dark_sky.data = data["dark_maps"].copy()
        data.close()

    dark_sky_data = np.empty(hp.nside2npix(nside), dtype=dark_sky.data.dtype)

    for band in dark_sky_data.dtype.names:
        dark_sky_data[band] = hp.pixelfunc.ud_grade(dark_sky.data[band], nside_out=nside)

    return dark_sky_data


def dark_m5(decs, bandname, latitude_rad, fiducial_FWHMEff, exptime=30.0, nexp=1):
    """Return a nominal best-depth map of the sky

    Parameters
    ----------
    decs : `float`
        The declinations for the desired points. Float or adday-like. (radians)
    bandname : `str`
        Name of band, one of ugrizy.
    latitude_rad : `float`
        Latitude of the observatory (radians)
    fiducial_FWHMEff : `float`
        The fiducial seeing FWHMeff to use (arcsec).
    exptime : `float`
        The fiducial exposure time to assume (seconds). Default 30.
    nexp : `int`
        The number of exposures per visit. Default 1.


    """
    nside = hp.npix2nside(np.size(decs))
    ds = dark_sky(nside)[bandname]
    min_z = np.abs(decs - latitude_rad)
    airmass_min = 1 / np.cos(min_z)
    airmass_min = np.where(airmass_min < 0, np.nan, airmass_min)
    sm = SeeingModel(band_list=[bandname])
    fwhm_eff = sm(fiducial_FWHMEff, airmass_min)["fwhmEff"][0]
    dark_m5_map = m5_flat_sed(
        bandname,
        musky=ds,
        fwhm_eff=fwhm_eff,
        exp_time=exptime,
        airmass=airmass_min,
        nexp=nexp,
        tau_cloud=0,
    )
    return dark_m5_map
