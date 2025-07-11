__all__ = ("generate_ddf_grid",)

import argparse
import os
import sys

import astropy.units as u
import numpy as np
from astropy.time import Time

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.site_models.seeing_model import SeeingModel
from rubin_scheduler.utils import SURVEY_START_MJD, Site, ddf_locations, m5_flat_sed


def generate_ddf_grid(
    verbose=True,
    mjd0=59560.2,
    delta_t=15.0,
    survey_length=40.0,
    sun_limit=-12,
    nominal_expt=30.0,
):
    """Pre-compute conditions for DDF locations over survey

    Parameters
    ----------
    mjd0 : `float`
        The start MJD of the grid
    delta_t : `float`
        Spacing of time steps in minutes. Default 15
    survey_length : `float`
        Full span of DDF grid (years). Default 40.
    sun_limit : `float`
        Ignore times with sun above sun limit in degrees.
        Default -12.
    nominal_expt : `float`
        Nominal exposure time in seconds to use for depth visits.
        Default 30
    """

    # Technically this script should be over in rubin_sim, but here to be more
    # easily found. Bury import here so it's hopefully not a problem.
    import rubin_sim.skybrightness as sb
    from astroplan import Observer

    dds = ddf_locations()
    delta_t = delta_t / 60.0 / 24.0  # to days
    survey_length = survey_length * 365.25
    sun_limit = np.radians(sun_limit)  # degrees

    nominal_seeing = 0.7  # arcsec

    site = Site("LSST")
    observer = Observer(
        longitude=site.longitude * u.deg,
        latitude=site.latitude * u.deg,
        elevation=site.height * u.m,
        name="LSST",
    )

    seeing_model = SeeingModel()

    seeing_indx = 1  # 0=u, 1=g, 2=r, etc.

    mjds = np.arange(mjd0, mjd0 + survey_length, delta_t)

    names = ["mjd", "sun_alt", "sun_n18_rising_next", "moon_phase", "moon_alt"]
    for survey_name in dds.keys():
        names.append(survey_name + "_airmass")
        names.append(survey_name + "_sky_g")
        names.append(survey_name + "_m5_g")

    types = [float] * len(names)
    result = np.zeros(mjds.size, dtype=list(zip(names, types)))
    result["mjd"] = mjds

    # pretty sure these are radians
    ras = np.radians(np.array([dds[survey][0] for survey in dds]))
    decs = np.radians(np.array([dds[survey][1] for survey in dds]))

    sm = sb.SkyModel(mags=True)
    mags = []
    airmasses = []
    sun_alts = []
    moon_alts = []
    moon_phases = []

    maxi = mjds.size
    for i, mjd in enumerate(mjds):
        if verbose:
            progress = i / maxi * 100
            text = "\rprogress = %0.1f%%" % progress
            sys.stdout.write(text)
            sys.stdout.flush()

        try:
            sm.set_ra_dec_mjd(ras, decs, mjd, degrees=False)
        except ValueError:
            sm.sun_alt = 12.0
        if sm.sun_alt > sun_limit:
            mags.append(sm.return_mags()["g"] * 0)
            airmasses.append(sm.airmass * 0)
        else:
            mags.append(sm.return_mags()["g"])
            airmasses.append(sm.airmass)
        sun_alts.append(sm.sun_alt)
        result["sun_n18_rising_next"][i] = observer.twilight_morning_astronomical(
            Time(mjd, format="mjd"), which="next"
        ).mjd
        moon_alts.append(sm.moon_alt)
        moon_phases.append(sm.moon_phase)

    mags = np.array(mags)
    airmasses = np.array(airmasses)
    result["sun_alt"] = sun_alts
    result["moon_alt"] = moon_alts
    result["moon_phase"] = moon_phases

    for i, survey_name in enumerate(dds.keys()):
        result[survey_name + "_airmass"] = airmasses[:, i]
        result[survey_name + "_sky_g"] = mags[:, i]

        # now to compute the expected seeing if the zenith is nominal
        FWHMeff = seeing_model(nominal_seeing, airmasses[:, i])["fwhmEff"][seeing_indx, :]
        result[survey_name + "_m5_g"] = m5_flat_sed(
            "g", mags[:, i], FWHMeff, nominal_expt, airmasses[:, i], nexp=1
        )

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        dest="full",
        action="store_true",
        help="Compute 40 year grid at 15 min resolution. Default is 12 years at 5 min resolution.",
    )
    parser.set_defaults(full=False)

    args = parser.parse_args()

    full = args.full

    if full:
        result = generate_ddf_grid()
        np.savez(os.path.join(get_data_dir(), "scheduler", "ddf_grid.npz"), ddf_grid=result)
    else:
        result = generate_ddf_grid(verbose=True, mjd0=SURVEY_START_MJD - 365, delta_t=5.0, survey_length=12.0)
        np.savez(os.path.join(get_data_dir(), "scheduler", "ddf_grid_fine.npz"), ddf_grid=result)
