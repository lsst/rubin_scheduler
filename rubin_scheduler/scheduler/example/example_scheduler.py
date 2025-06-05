__all__ = (
    "example_scheduler",
    "sched_argparser",
    "set_run_info",
    "run_sched",
    "gen_long_gaps_survey",
    "gen_greedy_surveys",
    "generate_blobs",
    "generate_twi_blobs",
    "generate_twilight_near_sun",
    "standard_bf",
)

import argparse
import os
import subprocess
import sys

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils import iers

import rubin_scheduler
import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler, SimpleBandSched
from rubin_scheduler.scheduler.surveys import (
    BlobSurvey,
    GreedySurvey,
    LongGapSurvey,
    ScriptedSurvey,
    gen_roman_off_season,
    gen_roman_on_season,
    gen_too_surveys,
    generate_ddf_scheduled_obs,
)
from rubin_scheduler.scheduler.targetofo import gen_all_events
from rubin_scheduler.scheduler.utils import ConstantFootprint, CurrentAreaMap, make_rolling_footprints
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD, _hpid2_ra_dec

# So things don't fail on hyak
iers.conf.auto_download = False
# XXX--note this line probably shouldn't be in production
iers.conf.auto_max_age = None


def example_scheduler(
    nside: int = DEFAULT_NSIDE,
    mjd_start: float = SURVEY_START_MJD,
    no_too: bool = False,
) -> CoreScheduler:
    """Provide an example baseline survey-strategy scheduler.

    Parameters
    ----------
    nside : `int`
        Nside for the scheduler maps and basis functions.
    mjd_start : `float`
        Start date for the survey (MJD).
    no_too : `bool`
        Turn off ToO simulation. Default False.

    Returns
    -------
    scheduler : `rubin_scheduler.scheduler.CoreScheduler`
        A scheduler set up as the baseline survey strategy.
    """
    parser = sched_argparser()
    args = parser.parse_args(args=[])
    args.setup_only = True
    args.no_too = no_too
    args.dbroot = "example_"
    args.outDir = "."
    args.nside = nside
    args.mjd_start = mjd_start
    scheduler = gen_scheduler(args)
    return scheduler


def standard_bf(
    nside,
    bandname="g",
    bandname2="i",
    m5_weight=6.0,
    footprint_weight=1.5,
    slewtime_weight=3.0,
    stayband_weight=3.0,
    template_weight=12.0,
    u_template_weight=50.0,
    g_template_weight=50.0,
    footprints=None,
    n_obs_template=None,
    season=365.25,
    season_start_hour=-4.0,
    season_end_hour=2.0,
    moon_distance=30.0,
    strict=True,
    wind_speed_maximum=20.0,
):
    """Generate the standard basis functions that are shared by blob surveys

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use. Defaults to DEFAULT_NSIDE
    bandname : `str`
        The band name for the first observation. Default "g".
    bandname2 : `str`
        The band name for the second in the pair (None if unpaired).
        Default "i".
    n_obs_template : `dict`
        The number of observations to take every season in each band.
        Default None.
    season : `float`
        The length of season (i.e., how long before templates expire) (days).
        Default 365.25.
    season_start_hour : `float`
        Hour angle limits to use when gathering templates.
        Default -4 (hours)
    sesason_end_hour : `float`
       Hour angle limits to use when gathering templates.
       Default +2 (hours)
    moon_distance : `float`
        The mask radius to apply around the moon (degrees).
        Default 30.
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
        Default 6.0 (unitless)
    footprint_weight : `float`
        The weight on the survey footprint basis function.
        Default 0.3 (unitless)
    slewtime_weight : `float`
        The weight on the slewtime basis function. Default 3 (unitless).
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
        Default 3 (unitless).
    template_weight : `float`
        The weight to place on getting image templates every season.
        Default 12 (unitless).
    u_template_weight : `float`
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a little
        higher than the standard template_weight kwarg.
        Default 24 (unitless)
    g_template_weight : `float`
        The weight to place on getting image templates in g-band. Since there
        are so few g-visits, it can be helpful to turn this up a
        little higher than the standard template_weight kwarg.
        Default 24 (unitless).

    Returns
    -------
    basis_functions_weights : `list`
        list of tuple pairs (basis function, weight) that is
        (rubin_scheduler.scheduler.BasisFunction object, float)

    """
    template_weights = {
        "u": u_template_weight,
        "g": g_template_weight,
        "r": template_weight,
        "i": template_weight,
        "z": template_weight,
        "y": template_weight,
    }

    bfs = []

    if bandname2 is not None:
        bfs.append(
            (
                bf.M5DiffBasisFunction(bandname=bandname, nside=nside),
                m5_weight / 2.0,
            )
        )
        bfs.append(
            (
                bf.M5DiffBasisFunction(bandname=bandname2, nside=nside),
                m5_weight / 2.0,
            )
        )

    else:
        bfs.append((bf.M5DiffBasisFunction(bandname=bandname, nside=nside), m5_weight))

    if bandname2 is not None:
        bfs.append(
            (
                bf.FootprintBasisFunction(
                    bandname=bandname,
                    footprint=footprints,
                    out_of_bounds_val=np.nan,
                    nside=nside,
                ),
                footprint_weight / 2.0,
            )
        )
        bfs.append(
            (
                bf.FootprintBasisFunction(
                    bandname=bandname2,
                    footprint=footprints,
                    out_of_bounds_val=np.nan,
                    nside=nside,
                ),
                footprint_weight / 2.0,
            )
        )
    else:
        bfs.append(
            (
                bf.FootprintBasisFunction(
                    bandname=bandname,
                    footprint=footprints,
                    out_of_bounds_val=np.nan,
                    nside=nside,
                ),
                footprint_weight,
            )
        )

    bfs.append(
        (
            bf.SlewtimeBasisFunction(bandname=bandname, nside=nside),
            slewtime_weight,
        )
    )
    if strict:
        bfs.append((bf.StrictBandBasisFunction(bandname=bandname), stayband_weight))
    else:
        bfs.append((bf.BandChangeBasisFunction(bandname=bandname), stayband_weight))

    if n_obs_template is not None:
        if bandname2 is not None:
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        bandname=bandname,
                        nside=nside,
                        footprint=footprints.get_footprint(bandname),
                        n_obs=n_obs_template[bandname],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weights[bandname] / 2.0,
                )
            )
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        bandname=bandname2,
                        nside=nside,
                        footprint=footprints.get_footprint(bandname2),
                        n_obs=n_obs_template[bandname2],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weights[bandname2] / 2.0,
                )
            )
        else:
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        bandname=bandname,
                        nside=nside,
                        footprint=footprints.get_footprint(bandname),
                        n_obs=n_obs_template[bandname],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weights[bandname],
                )
            )

    # The shared masks
    bfs.append(
        (
            bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance),
            0.0,
        )
    )
    bfs.append((bf.AvoidDirectWind(nside=nside, wind_speed_maximum=wind_speed_maximum), 0))
    bandnames = [fn for fn in [bandname, bandname2] if fn is not None]
    bfs.append((bf.BandLoadedBasisFunction(bandnames=bandnames), 0))
    bfs.append((bf.PlanetMaskBasisFunction(nside=nside), 0.0))

    return bfs


def blob_for_long(
    nside,
    nexp=2,
    exptime=29.2,
    band1s=["g"],
    band2s=["i"],
    pair_time=33.0,
    camera_rot_limits=[-80.0, 80.0],
    n_obs_template=None,
    season=365.25,
    season_start_hour=-4.0,
    season_end_hour=2.0,
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs=["DD", "twilight_near_sun"],
    m5_weight=6.0,
    footprint_weight=1.5,
    slewtime_weight=3.0,
    stayband_weight=3.0,
    template_weight=12.0,
    u_template_weight=50.0,
    g_template_weight=50.0,
    footprints=None,
    u_nexp1=True,
    night_pattern=[True, True],
    time_after_twi=30.0,
    HA_min=12,
    HA_max=24 - 3.5,
    blob_names=[],
    u_exptime=38.0,
    scheduled_respect=30.0,
):
    """
    Generate surveys that take observations in blobs.

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use. Default to DEFAULT_NSIDE.
    nexp : `int`
        The number of exposures to use in a visit. Default 2.
    exptime : `float`
        The exposure time to use per visit (seconds).
        Default 29.2
    band1s : `list` [`str`]
        The bandnames for the first band in a pair.
        Default ["g"].
    band2s : `list` of `str`
        The band names for the second in the pair (None if unpaired).
        Default ["i"].
    pair_time : `float`
        The ideal time between pairs (minutes). Default 33.
    camera_rot_limits : `list` of `float`
        The limits to impose when rotationally dithering the camera (degrees).
        Default [-80., 80.].
    n_obs_template : `dict`
        The number of observations to take every season in each band.
        If None, sets to 3 each. Default None.
    season : float
        The length of season (i.e., how long before templates expire) (days)
        Default 365.25.
    season_start_hour : `float`
        Hour angle limits to use when gathering templates.
        Default -4 (hours)
    sesason_end_hour : `float`
       Hour angle limits to use when gathering templates.
       Default +2 (hours)
    shadow_minutes : `float`
        Used to mask regions around zenith (minutes). Default 60.
    max_alt : `float`
        The maximium altitude to use when masking zenith (degrees).
        Default 76.
    moon_distance : `float`
        The mask radius to apply around the moon (degrees).
        Default 30.
    ignore_obs : `str` or `list` of `str`
        Ignore observations by surveys that include the given substring(s).
        Default "DD".
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
        Default 3 (unitless).
    footprint_weight : `float`
        The weight on the survey footprint basis function.
        Default 0.3 (uniteless).
    slewtime_weight : `float`
        The weight on the slewtime basis function.
        Default 3.0 (uniteless).
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
        Default 3.0 (uniteless).
    template_weight : `float`
        The weight to place on getting image templates every season.
        Default 12 (uniteless).
    u_template_weight : `float`
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a
        little higher than the standard template_weight kwarg.
        Default 24 (unitless.)
    u_nexp1 : `bool`
        Add a detailer to make sure the number of expossures
        in a visit is always 1 for u observations.
        Default True.
    """

    BlobSurvey_params = {
        "slew_approx": 7.5,
        "band_change_approx": 140.0,
        "read_approx": 2.0,
        "flush_time": 30.0,
        "smoothing_kernel": None,
        "nside": nside,
        "seed": 42,
        "dither": True,
        "twilight_scale": True,
    }

    surveys = []
    if n_obs_template is None:
        n_obs_template = {"u": 3, "g": 3, "r": 3, "i": 3, "z": 3, "y": 3}

    times_needed = [pair_time, pair_time * 2]
    for bandname, bandname2 in zip(band1s, band2s):
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
        )
        detailer_list.append(detailers.CloseAltDetailer())
        detailer_list.append(detailers.BandNexp(bandname="u", nexp=1, exptime=u_exptime))
        detailer_list.append(detailers.LabelRegionsAndDDFs())

        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        bfs.extend(
            standard_bf(
                nside,
                bandname=bandname,
                bandname2=bandname2,
                m5_weight=m5_weight,
                footprint_weight=footprint_weight,
                slewtime_weight=slewtime_weight,
                stayband_weight=stayband_weight,
                template_weight=template_weight,
                u_template_weight=u_template_weight,
                g_template_weight=g_template_weight,
                footprints=footprints,
                n_obs_template=n_obs_template,
                season=season,
                season_start_hour=season_start_hour,
                season_end_hour=season_end_hour,
            )
        )

        # Make sure we respect scheduled observations
        bfs.append((bf.TimeToScheduledBasisFunction(time_needed=scheduled_respect), 0))

        # Masks, give these 0 weight
        bfs.append(
            (
                bf.AltAzShadowMaskBasisFunction(
                    nside=nside, shadow_minutes=shadow_minutes, max_alt=max_alt, pad=3.0
                ),
                0.0,
            )
        )
        if bandname2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.TimeToTwilightBasisFunction(time_needed=time_needed), 0.0))
        bfs.append((bf.NotTwilightBasisFunction(), 0.0))
        bfs.append((bf.AfterEveningTwiBasisFunction(time_after=time_after_twi), 0.0))
        bfs.append((bf.HaMaskBasisFunction(ha_min=HA_min, ha_max=HA_max, nside=nside), 0.0))
        # don't execute every night
        bfs.append((bf.NightModuloBasisFunction(night_pattern), 0.0))
        # only execute one blob per night
        bfs.append((bf.OnceInNightBasisFunction(notes=blob_names), 0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if bandname2 is None:
            survey_name = "blob_long, %s" % bandname
        else:
            survey_name = "blob_long, %s%s" % (bandname, bandname2)
        if bandname2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(bandname=bandname2))

        if u_nexp1:
            detailer_list.append(detailers.BandNexp(bandname="u", nexp=1))
        surveys.append(
            BlobSurvey(
                basis_functions,
                weights,
                bandname1=bandname,
                bandname2=bandname2,
                exptime=exptime,
                ideal_pair_time=pair_time,
                survey_name=survey_name,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                **BlobSurvey_params,
            )
        )

    return surveys


def gen_long_gaps_survey(
    footprints,
    nside=DEFAULT_NSIDE,
    night_pattern=[True, True],
    gap_range=[2, 7],
    HA_min=12,
    HA_max=24 - 3.5,
    time_after_twi=120,
    u_template_weight=50.0,
    g_template_weight=50.0,
    u_exptime=38.0,
    nexp=2,
):
    """
    Paramterers
    -----------
    footprints : `rubin_scheduler.scheduler.utils.footprints.Footprints`
        The footprints to be used.
    night_pattern : `list` [`bool`]
        Which nights to let the survey execute. Default of [True, True]
        executes every night.
    gap_range : `list` of `float`
        Range of times to attempt to to gather pairs (hours).
        Default [2, 7].
    HA_min : `float`
        The hour angle limits passed to the initial blob scheduler.
        Default 12 (hours)
    HA_max : `float`
        The hour angle limits passed to the initial blob scheduler.
        Default 20.5 (hours).
    time_after_twi : `float`
        The time after evening twilight to attempt long gaps (minutes).
        Default 120.
    u_template_weight : `float`
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a
        little higher than the standard template_weight kwarg.
        Default 50 (unitless.)
    g_template_weight : `float`
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a
        little higher than the standard template_weight kwarg.
        Default 50 (unitless.)
    u_exptime : `float`
        Exposure time to use for u-band visits (seconds).
        Default 38.
    nexp : `int`
        Number of exposures per visit. Default 2.
    """

    surveys = []
    f1 = ["g", "r", "i"]
    f2 = ["r", "i", "z"]
    # Maybe force scripted to not go in twilight?
    blob_names = []
    for fn1, fn2 in zip(f1, f2):
        for ab in ["a", "b"]:
            blob_names.append("blob_long, %s%s, %s" % (fn1, fn2, ab))
    for bandname1, bandname2 in zip(f1, f2):
        blob = blob_for_long(
            footprints=footprints,
            nside=nside,
            band1s=[bandname1],
            band2s=[bandname2],
            night_pattern=night_pattern,
            time_after_twi=time_after_twi,
            HA_min=HA_min,
            HA_max=HA_max,
            u_template_weight=u_template_weight,
            g_template_weight=g_template_weight,
            blob_names=blob_names,
            u_exptime=u_exptime,
            nexp=nexp,
        )
        scripted = ScriptedSurvey(
            [bf.AvoidDirectWind(nside=nside)],
            nside=nside,
            ignore_obs=["blob", "DDF", "twi", "pair"],
        )
        surveys.append(LongGapSurvey(blob[0], scripted, gap_range=gap_range, avoid_zenith=True))

    return surveys


def gen_greedy_surveys(
    nside=DEFAULT_NSIDE,
    nexp=2,
    exptime=29.2,
    bands=["r", "i", "z", "y"],
    camera_rot_limits=[-80.0, 80.0],
    shadow_minutes=0.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs=["DD", "twilight_near_sun"],
    m5_weight=3.0,
    footprint_weight=0.75,
    slewtime_weight=3.0,
    stayband_weight=100.0,
    repeat_weight=-1.0,
    footprints=None,
):
    """
    Make a quick set of greedy surveys

    This is a convenience function to generate a list of survey objects
    that can be used with
    rubin_scheduler.scheduler.schedulers.Core_scheduler.
    To ensure we are robust against changes in the sims_featureScheduler
    codebase, all kwargs are
    explicitly set.

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use
    nexp : `int`
        The number of exposures to use in a visit. Default 1.
    exptime : `float`
        The exposure time to use per visit (seconds).
        Default 29.2
    bands : `list` of `str`
        Which bands to generate surveys for.
        Default ['r', 'i', 'z', 'y'].
    camera_rot_limits : `list` of `float`
        The limits to impose when rotationally dithering the camera (degrees).
        Default [-80., 80.].
    shadow_minutes : `float`
        Used to mask regions around zenith (minutes).
        Default 60.
    max_alt : `float`
        The maximium altitude to use when masking zenith (degrees).
        Default 76
    moon_distance : `float`
        The mask radius to apply around the moon (degrees).
        Default 30.
    ignore_obs : `str` or `list` of `str`
        Ignore observations by surveys that include the given substring(s).
        Default ["DD", "twilight_near_sun"].
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
        Default 3 (unitless).
    footprint_weight : `float`
        The weight on the survey footprint basis function.
        Default 0.3 (uniteless).
    slewtime_weight : `float`
        The weight on the slewtime basis function.
        Default 3.0 (uniteless).
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
        Default 3.0 (uniteless).
    """
    # Define the extra parameters that are used in the greedy survey. I
    # think these are fairly set, so no need to promote to utility func kwargs
    greed_survey_params = {
        "block_size": 1,
        "smoothing_kernel": None,
        "seed": 42,
        "camera": "LSST",
        "dither": True,
        "survey_name": "greedy",
    }

    surveys = []
    detailer_list = [
        detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
    ]
    detailer_list.append(detailers.Rottep2RotspDesiredDetailer())
    detailer_list.append(detailers.LabelRegionsAndDDFs())

    for bandname in bands:
        bfs = []
        bfs.extend(
            standard_bf(
                nside,
                bandname=bandname,
                bandname2=None,
                m5_weight=m5_weight,
                footprint_weight=footprint_weight,
                slewtime_weight=slewtime_weight,
                stayband_weight=stayband_weight,
                template_weight=0,
                u_template_weight=0,
                g_template_weight=0,
                footprints=footprints,
                n_obs_template=None,
                strict=False,
            )
        )

        bfs.append(
            (
                bf.VisitRepeatBasisFunction(
                    gap_min=0, gap_max=2 * 60.0, bandname=None, nside=nside, npairs=20
                ),
                repeat_weight,
            )
        )
        # Masks, give these 0 weight
        bfs.append(
            (
                bf.AltAzShadowMaskBasisFunction(
                    nside=nside, shadow_minutes=shadow_minutes, max_alt=max_alt, pad=3.0
                ),
                0,
            )
        )
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        surveys.append(
            GreedySurvey(
                basis_functions,
                weights,
                exptime=exptime,
                bandname=bandname,
                nside=nside,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                **greed_survey_params,
            )
        )

    return surveys


def generate_blobs(
    nside,
    nexp=2,
    exptime=29.2,
    band1s=["u", "u", "g", "r", "i", "z", "y"],
    band2s=["g", "r", "r", "i", "z", "y", "y"],
    pair_time=33.0,
    camera_rot_limits=[-80.0, 80.0],
    n_obs_template=None,
    season=365.25,
    season_start_hour=-4.0,
    season_end_hour=2.0,
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs=["DD", "twilight_near_sun"],
    m5_weight=6.0,
    footprint_weight=1.5,
    slewtime_weight=3.0,
    stayband_weight=3.0,
    template_weight=12.0,
    u_template_weight=50.0,
    g_template_weight=50.0,
    footprints=None,
    u_nexp1=True,
    scheduled_respect=45.0,
    good_seeing={"g": 3, "r": 3, "i": 3},
    good_seeing_weight=3.0,
    mjd_start=1,
    repeat_weight=-20,
    u_exptime=38.0,
):
    """
    Generate surveys that take observations in blobs.

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use
    nexp : int
        The number of exposures to use in a visit.
        Default 1.
    exptime : `float`
        The exposure time to use per visit (seconds).
        Default 29.2
    band1s : `list` [`str`]
        The bandnames for the first set.
        Default ["u", "u", "g", "r", "i", "z", "y"]
    band2s : `list` of `str`
        The band names for the second in the pair (None if unpaired)
        Default ["g", "r", "r", "i", "z", "y", "y"].
    pair_time : `float`
        The ideal time between pairs (minutes).
        Default 33.
    camera_rot_limits : `list` of `float`
        The limits to impose when rotationally dithering the camera (degrees).
        Default [-80., 80.].
    n_obs_template : `dict`
        The number of observations to take every season in each band.
        If None, sets to 3 each.
    season : `float`
        The length of season (i.e., how long before templates expire) (days).
        Default 365.25.
    season_start_hour : `float`
        Hour angle limits to use when gathering templates.
        Default -4 (hours)
    sesason_end_hour : `float`
       Hour angle limits to use when gathering templates.
       Default +2 (hours)
    shadow_minutes : `float`
        Used to mask regions around zenith (minutes).
        Default 60.
    max_alt : `float`
        The maximium altitude to use when masking zenith (degrees).
        Default 76.
    moon_distance : `float`
        The mask radius to apply around the moon (degrees).
        Default 30.
    ignore_obs : `str` or `list` of `str`
        Ignore observations by surveys that include the given substring(s).
        Default ["DD", "twilight_near_sun"].
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
        Default 3 (unitless).
    footprint_weight : `float`
        The weight on the survey footprint basis function.
        Default 0.3 (uniteless).
    slewtime_weight : `float`
        The weight on the slewtime basis function.
        Default 3.0 (uniteless).
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
        Default 3.0 (uniteless).
    template_weight : `float`
        The weight to place on getting image templates every season.
        Default 12 (unitless).
    u_template_weight : `float`
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a
        little higher than the standard template_weight kwarg.
        Default 24 (unitless).
    u_nexp1 : `bool`
        Add a detailer to make sure the number of expossures in a visit
        is always 1 for u observations. Default True.
    scheduled_respect : `float`
        How much time to require there be before a pre-scheduled
        observation (minutes). Default 45.
    """

    BlobSurvey_params = {
        "slew_approx": 7.5,
        "band_change_approx": 140.0,
        "read_approx": 2.0,
        "flush_time": 30.0,
        "smoothing_kernel": None,
        "nside": nside,
        "seed": 42,
        "dither": True,
        "twilight_scale": False,
    }

    if n_obs_template is None:
        n_obs_template = {"u": 3, "g": 3, "r": 3, "i": 3, "z": 3, "y": 3}

    surveys = []

    times_needed = [pair_time, pair_time * 2]
    for bandname, bandname2 in zip(band1s, band2s):
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
        )
        detailer_list.append(detailers.Rottep2RotspDesiredDetailer())
        detailer_list.append(detailers.CloseAltDetailer())
        detailer_list.append(detailers.FlushForSchedDetailer())
        detailer_list.append(detailers.LabelRegionsAndDDFs())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        bfs.extend(
            standard_bf(
                nside,
                bandname=bandname,
                bandname2=bandname2,
                m5_weight=m5_weight,
                footprint_weight=footprint_weight,
                slewtime_weight=slewtime_weight,
                stayband_weight=stayband_weight,
                template_weight=template_weight,
                u_template_weight=u_template_weight,
                g_template_weight=g_template_weight,
                footprints=footprints,
                n_obs_template=n_obs_template,
                season=season,
                season_start_hour=season_start_hour,
                season_end_hour=season_end_hour,
            )
        )

        bfs.append(
            (
                bf.VisitRepeatBasisFunction(
                    gap_min=0, gap_max=3 * 60.0, bandname=None, nside=nside, npairs=20
                ),
                repeat_weight,
            )
        )

        # Insert things for getting good seeing templates
        if bandname2 is not None:
            if bandname in list(good_seeing.keys()):
                bfs.append(
                    (
                        bf.NGoodSeeingBasisFunction(
                            bandname=bandname,
                            nside=nside,
                            mjd_start=mjd_start,
                            footprint=footprints.get_footprint(bandname),
                            n_obs_desired=good_seeing[bandname],
                        ),
                        good_seeing_weight,
                    )
                )
            if bandname2 in list(good_seeing.keys()):
                bfs.append(
                    (
                        bf.NGoodSeeingBasisFunction(
                            bandname=bandname2,
                            nside=nside,
                            mjd_start=mjd_start,
                            footprint=footprints.get_footprint(bandname2),
                            n_obs_desired=good_seeing[bandname2],
                        ),
                        good_seeing_weight,
                    )
                )
        else:
            if bandname in list(good_seeing.keys()):
                bfs.append(
                    (
                        bf.NGoodSeeingBasisFunction(
                            bandname=bandname,
                            nside=nside,
                            mjd_start=mjd_start,
                            footprint=footprints.get_footprint(bandname),
                            n_obs_desired=good_seeing[bandname],
                        ),
                        good_seeing_weight,
                    )
                )
        # Make sure we respect scheduled observations
        bfs.append((bf.TimeToScheduledBasisFunction(time_needed=scheduled_respect), 0))
        # Masks, give these 0 weight
        bfs.append(
            (
                bf.AltAzShadowMaskBasisFunction(
                    nside=nside, shadow_minutes=shadow_minutes, max_alt=max_alt, pad=3.0
                ),
                0.0,
            )
        )
        if bandname2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.TimeToTwilightBasisFunction(time_needed=time_needed), 0.0))
        bfs.append((bf.NotTwilightBasisFunction(), 0.0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if bandname2 is None:
            survey_name = "pair_%i, %s" % (pair_time, bandname)
        else:
            survey_name = "pair_%i, %s%s" % (pair_time, bandname, bandname2)
        if bandname2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(bandname=bandname2))

        if u_nexp1:
            detailer_list.append(detailers.BandNexp(bandname="u", nexp=1, exptime=u_exptime))
        surveys.append(
            BlobSurvey(
                basis_functions,
                weights,
                bandname1=bandname,
                bandname2=bandname2,
                exptime=exptime,
                ideal_pair_time=pair_time,
                survey_name=survey_name,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                **BlobSurvey_params,
            )
        )

    return surveys


def generate_twi_blobs(
    nside,
    nexp=2,
    exptime=29.2,
    band1s=["r", "i", "z", "y"],
    band2s=["i", "z", "y", "y"],
    pair_time=15.0,
    camera_rot_limits=[-80.0, 80.0],
    n_obs_template=None,
    season=365.25,
    season_start_hour=-4.0,
    season_end_hour=2.0,
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs=["DD", "twilight_near_sun"],
    m5_weight=6.0,
    footprint_weight=1.5,
    slewtime_weight=3.0,
    stayband_weight=3.0,
    template_weight=12.0,
    footprints=None,
    repeat_night_weight=None,
    wfd_footprint=None,
    scheduled_respect=15.0,
    repeat_weight=-1.0,
    night_pattern=None,
):
    """
    Generate surveys that take observations in blobs.

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use
    nexp : `int`
        The number of exposures to use in a visit.
    exptime : `float`
        The exposure time to use per visit (seconds).
        Default 29.2
    band1s : `list` of `str`
        The bandnames for the first set.
        Default ["r", "i", "z", "y"].
    band2s : `list` of `str`
        The band names for the second in the pair (None if unpaired).
        Default ["i", "z", "y", "y"].
    pair_time : `float`
        The ideal time between pairs (minutes). Default 22.
    camera_rot_limits : `list` of `float`
        The limits to impose when rotationally dithering the camera (degrees).
        Default [-80., 80.].
    n_obs_template : `dict`
        The number of observations to take every season in each band.
        If None, sets to 3 each. Default None.
    season : `float`
        The length of season (i.e., how long before templates expire) (days).
        Default 365.25.
    season_start_hour : `float`
        Hour angle limits to use when gathering templates.
        Default -4 (hours)
    sesason_end_hour : `float`
       Hour angle limits to use when gathering templates.
       Default +2 (hours)
    shadow_minutes : `float`
        Used to mask regions around zenith (minutes).
        Default 60.
    max_alt : `float`
        The maximium altitude to use when masking zenith (degrees).
        Default 76.
    moon_distance : `float`
        The mask radius to apply around the moon (degrees).
        Default 30.
    ignore_obs : `str` or `list` of `str`
        Ignore observations by surveys that include the given substring(s).
        Default ["DD", "twilight_near_sun"].
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
        Default 3 (unitless).
    footprint_weight : `float`
        The weight on the survey footprint basis function.
        Default 0.3 (uniteless).
    slewtime_weight : `float`
        The weight on the slewtime basis function.
        Default 3.0 (uniteless).
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
        Default 3.0 (uniteless).
    template_weight : `float`
        The weight to place on getting image templates every season.
        Default 12 (unitless).
    u_template_weight : `float`
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a
        little higher than the standard template_weight kwarg.
        Default 24 (unitless).
    """

    BlobSurvey_params = {
        "slew_approx": 7.5,
        "band_change_approx": 140.0,
        "read_approx": 2.0,
        "flush_time": 30.0,
        "smoothing_kernel": None,
        "nside": nside,
        "seed": 42,
        "dither": True,
        "twilight_scale": False,
        "in_twilight": True,
    }

    surveys = []

    if n_obs_template is None:
        n_obs_template = {"u": 3, "g": 3, "r": 3, "i": 3, "z": 3, "y": 3}

    times_needed = [pair_time, pair_time * 2]
    for bandname, bandname2 in zip(band1s, band2s):
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
        )
        detailer_list.append(detailers.Rottep2RotspDesiredDetailer())
        detailer_list.append(detailers.CloseAltDetailer())
        detailer_list.append(detailers.FlushForSchedDetailer())
        detailer_list.append(detailers.LabelRegionsAndDDFs())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        bfs.extend(
            standard_bf(
                nside,
                bandname=bandname,
                bandname2=bandname2,
                m5_weight=m5_weight,
                footprint_weight=footprint_weight,
                slewtime_weight=slewtime_weight,
                stayband_weight=stayband_weight,
                template_weight=template_weight,
                u_template_weight=0,
                g_template_weight=0,
                footprints=footprints,
                n_obs_template=n_obs_template,
                season=season,
                season_start_hour=season_start_hour,
                season_end_hour=season_end_hour,
            )
        )

        bfs.append(
            (
                bf.VisitRepeatBasisFunction(
                    gap_min=0, gap_max=2 * 60.0, bandname=None, nside=nside, npairs=20
                ),
                repeat_weight,
            )
        )

        if repeat_night_weight is not None:
            bfs.append(
                (
                    bf.AvoidLongGapsBasisFunction(
                        nside=nside,
                        bandname=None,
                        min_gap=0.0,
                        max_gap=10.0 / 24.0,
                        ha_limit=3.5,
                        footprint=wfd_footprint,
                    ),
                    repeat_night_weight,
                )
            )
        # Make sure we respect scheduled observations
        bfs.append((bf.TimeToScheduledBasisFunction(time_needed=scheduled_respect), 0))
        # Masks, give these 0 weight
        bfs.append(
            (
                bf.AltAzShadowMaskBasisFunction(
                    nside=nside,
                    shadow_minutes=shadow_minutes,
                    max_alt=max_alt,
                    pad=3.0,
                ),
                0.0,
            )
        )
        if bandname2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.TimeToTwilightBasisFunction(time_needed=time_needed, alt_limit=12), 0.0))

        # Let's turn off twilight blobs on nights where we are
        # doing NEO hunts
        bfs.append((bf.NightModuloBasisFunction(pattern=night_pattern), 0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if bandname2 is None:
            survey_name = "pair_%i, %s" % (pair_time, bandname)
        else:
            survey_name = "pair_%i, %s%s" % (pair_time, bandname, bandname2)
        if bandname2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(bandname=bandname2))
        surveys.append(
            BlobSurvey(
                basis_functions,
                weights,
                bandname1=bandname,
                bandname2=bandname2,
                exptime=exptime,
                ideal_pair_time=pair_time,
                survey_name=survey_name,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                **BlobSurvey_params,
            )
        )

    return surveys


def ddf_surveys(
    detailers=None,
    offseason_length=73.05,
    euclid_detailers=None,
    nside=None,
    expt=29.2,
    nexp=2,
):
    """Generate surveys for DDF observations

    Parameters
    ----------
    detailers : `list` of `rubin_scheduler.scheduler.Detailer`
        Detailers for DDFs. Default None.
    season_unobs_frac : `float`
        Fraction of the season to not attempt DDF observations.
        Default 0.2.
    euclid_detailers : `list` of `rubin_scheduler.scheduler.Detailer`
        Detailers to use for Euclid DDF observations.
        Default None.
    expt : `float`
        Exposure time for DDF visits. Default 29.2.
    """
    nsnaps = [1, 2, 2, 2, 2, 2]
    if nexp == 1:
        nsnaps = [1, 1, 1, 1, 1, 1]
    obs_array = generate_ddf_scheduled_obs(offseason_length=offseason_length, expt=expt, nsnaps=nsnaps)
    euclid_obs = np.where(
        (obs_array["scheduler_note"] == "DD:EDFS_b") | (obs_array["scheduler_note"] == "DD:EDFS_a")
    )[0]
    all_other = np.where(
        (obs_array["scheduler_note"] != "DD:EDFS_b") & (obs_array["scheduler_note"] != "DD:EDFS_a")
    )[0]

    survey1 = ScriptedSurvey([bf.AvoidDirectWind(nside=nside)], nside=nside, detailers=detailers)
    survey1.set_script(obs_array[all_other])

    survey2 = ScriptedSurvey([bf.AvoidDirectWind(nside=nside)], nside=nside, detailers=euclid_detailers)
    survey2.set_script(obs_array[euclid_obs])

    return [survey1, survey2]


def ecliptic_target(nside=DEFAULT_NSIDE, dist_to_eclip=40.0, dec_max=30.0, mask=None):
    """Generate a target_map for the area around the ecliptic

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use
    dist_to_eclip : `float`
        The distance to the ecliptic to constrain to (degrees).
        Default 40.
    dec_max : `float`
        The max declination to alow (degrees).
        Default 30.
    mask : `np.array`
        Any additional mask to apply, should be a
        HEALpix mask with matching nside. Default None.
    """

    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
    result = np.zeros(ra.size)
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    eclip_lat = coord.barycentrictrueecliptic.lat.radian
    good = np.where((np.abs(eclip_lat) < np.radians(dist_to_eclip)) & (dec < np.radians(dec_max)))
    result[good] += 1

    if mask is not None:
        result *= mask

    return result


def generate_twilight_near_sun(
    nside,
    night_pattern=None,
    nexp=1,
    exptime=15,
    ideal_pair_time=5.0,
    max_airmass=2.0,
    camera_rot_limits=[-80.0, 80.0],
    time_needed=10,
    footprint_mask=None,
    footprint_weight=0.1,
    slewtime_weight=3.0,
    stayband_weight=3.0,
    min_area=None,
    bands="riz",
    n_repeat=4,
    sun_alt_limit=-14.8,
    slew_estimate=4.5,
    moon_distance=30.0,
    shadow_minutes=0,
    min_alt=20.0,
    max_alt=76.0,
    max_elong=60.0,
    ignore_obs=["DD", "pair", "long", "blob", "greedy"],
    band_dist_weight=0.3,
    time_to_12deg=25.0,
):
    """Generate a survey for observing NEO objects in twilight

    Parameters
    ----------
    night_pattern : `list` of `bool`
        A list of bools that set when the survey will be
        active. e.g., [True, False] for every-other night,
        [True, False, False] for every third night.
        Default None.
    nexp : `int`
        Number of snaps in a visit. Default 1.
    exptime : `float`
        Exposure time of visits. Default 15.
    ideal_pair_time : `float`
        Ideal time between repeat visits (minutes).
        Default 5
    max_airmass : `float`
        Maximum airmass to attempt (unitless). Default 2.
    camera_rot_limits : `list` of `float`
        The camera rotation limits to use (degrees).
        Default [-80, 80].
    time_needed : `float`
        How much time should be available
        (e.g., before twilight ends) (minutes).
        Default 10
    footprint_mask : `np.array`
        Mask to apply to the constructed ecliptic target mask (None).
        Default None
    footprint_weight : `float`
        Weight for footprint basis function. Default 0.1 (uniteless).
    slewtime_weight : `float`
        Weight for slewtime basis function. Default 3 (unitless)
    stayband_weight : `float`
        Weight for staying in the same band basis function.
        Default 3 (unitless)
    min_area : `float`
        The area that needs to be available before the survey will return
        observations (sq degrees?). Default None.
    bands : `str`
        The bands to use, default 'riz'
    n_repeat : `int`
        The number of times a blob should be repeated, default 4.
    sun_alt_limit : `float`
        Do not start unless sun is higher than this limit (degrees).
        Default -14.8.
    slew_estimate : `float`
        An estimate of how long it takes to slew between
        neighboring fields (seconds). Default 4.5
    time_to_sunrise : `float`
        Do not execute if time to sunrise is greater than (minutes).
        Default 25.
    """
    survey_name = "twilight_near_sun"
    footprint = ecliptic_target(nside=nside, mask=footprint_mask)
    constant_fp = ConstantFootprint(nside=nside)
    for bandname in bands:
        constant_fp.set_footprint(bandname, footprint)

    surveys = []
    for bandname in bands:
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
        )
        detailer_list.append(detailers.CloseAltDetailer())
        # Should put in a detailer so things start at lowest altitude
        detailer_list.append(detailers.TwilightTripleDetailer(slew_estimate=slew_estimate, n_repeat=n_repeat))
        detailer_list.append(detailers.RandomBandDetailer(bands=bands))
        detailer_list.append(detailers.LabelRegionsAndDDFs())
        bfs = []

        bfs.append(
            (
                bf.FootprintBasisFunction(
                    bandname=bandname,
                    footprint=constant_fp,
                    out_of_bounds_val=np.nan,
                    nside=nside,
                ),
                footprint_weight,
            )
        )

        bfs.append(
            (
                bf.SlewtimeBasisFunction(bandname=bandname, nside=nside),
                slewtime_weight,
            )
        )
        bfs.append((bf.StrictBandBasisFunction(bandname=bandname), stayband_weight))
        bfs.append((bf.BandDistBasisFunction(bandname=bandname), band_dist_weight))
        # Need a toward the sun, reward high airmass, with an
        # airmass cutoff basis function.
        bfs.append(
            (
                bf.NearSunHighAirmassBasisFunction(nside=nside, max_airmass=max_airmass),
                0,
            )
        )
        bfs.append(
            (
                bf.AltAzShadowMaskBasisFunction(
                    nside=nside,
                    shadow_minutes=shadow_minutes,
                    max_alt=max_alt,
                    min_alt=min_alt,
                    pad=3.0,
                ),
                0,
            )
        )
        bfs.append((bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance), 0))
        bfs.append((bf.BandLoadedBasisFunction(bandnames=bandname), 0))
        bfs.append((bf.PlanetMaskBasisFunction(nside=nside), 0))
        bfs.append(
            (
                bf.SolarElongationMaskBasisFunction(min_elong=0.0, max_elong=max_elong, nside=nside),
                0,
            )
        )

        bfs.append((bf.NightModuloBasisFunction(pattern=night_pattern), 0))
        # Do not attempt unless the sun is getting high
        bfs.append(
            (
                (
                    bf.CloseToTwilightBasisFunction(
                        max_sun_alt_limit=sun_alt_limit, max_time_to_12deg=time_to_12deg
                    )
                ),
                0,
            )
        )

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]

        # Set huge ideal pair time and use the detailer to cut down
        # the list of observations to fit twilight?
        surveys.append(
            BlobSurvey(
                basis_functions,
                weights,
                bandname1=bandname,
                bandname2=None,
                ideal_pair_time=ideal_pair_time,
                nside=nside,
                exptime=exptime,
                survey_name=survey_name,
                ignore_obs=ignore_obs,
                dither=True,
                nexp=nexp,
                detailers=detailer_list,
                twilight_scale=False,
                area_required=min_area,
            )
        )
    return surveys


def set_run_info(dbroot=None, file_end="v3.4_", out_dir="."):
    """Gather versions of software used to record"""
    extra_info = {}
    exec_command = ""
    for arg in sys.argv:
        exec_command += " " + arg
    extra_info["exec command"] = exec_command
    try:
        extra_info["git hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        extra_info["git hash"] = "Not in git repo"

    extra_info["file executed"] = os.path.realpath(__file__)
    try:
        rs_path = rubin_scheduler.__path__[0]
        hash_file = os.path.join(rs_path, "../", ".git/refs/heads/main")
        extra_info["rubin_scheduler git hash"] = subprocess.check_output(["cat", hash_file])
    except subprocess.CalledProcessError:
        pass

    # Use the filename of the script to name the output database
    if dbroot is None:
        fileroot = os.path.basename(sys.argv[0]).replace(".py", "") + "_"
    else:
        fileroot = dbroot + "_"
    fileroot = os.path.join(out_dir, fileroot + file_end)
    return fileroot, extra_info


def run_sched(
    scheduler,
    survey_length=365.25,
    nside=DEFAULT_NSIDE,
    filename=None,
    verbose=False,
    extra_info=None,
    illum_limit=40.0,
    mjd_start=60796.0,
    event_table=None,
    sim_to_o=None,
    snapshot_dir=None,
):
    """Run survey"""
    n_visit_limit = None
    fs = SimpleBandSched(illum_limit=illum_limit)
    observatory = ModelObservatory(nside=nside, mjd_start=mjd_start, sim_to_o=sim_to_o)
    observatory, scheduler, observations = sim_runner(
        observatory,
        scheduler,
        sim_duration=survey_length,
        filename=filename,
        delete_past=True,
        n_visit_limit=n_visit_limit,
        verbose=verbose,
        extra_info=extra_info,
        band_scheduler=fs,
        event_table=event_table,
        snapshot_dir=snapshot_dir,
    )

    return observatory, scheduler, observations


def gen_scheduler(args):
    survey_length = args.survey_length  # Days
    out_dir = args.out_dir
    verbose = args.verbose
    nexp = args.nexp
    dbroot = args.dbroot
    nside = args.nside
    mjd_plus = args.mjd_plus
    split_long = args.split_long
    snapshot_dir = args.snapshot_dir
    too = not args.no_too

    # Parameters that were previously command-line
    # arguments.
    max_dither = 0.2  # Degrees. For DDFs
    ddf_offseason_length = 365.25 * 0.2  # Amount of season not to use for DDFs
    illum_limit = 40.0  # Percent. Lunar illumination used for band loading
    u_exptime = 38.0  # Seconds
    nslice = 2  # N slices for rolling
    rolling_scale = 0.9  # Strength of rolling
    rolling_uniform = True  # Should we use the uniform rolling flag
    nights_off = 3  # For long gaps
    ei_night_pattern = 4  # select doing earth interior observation every 4 nights
    ei_bands = "riz"  # Bands to use for earth interior observations.
    ei_repeat = 4  # Number of times to repeat earth interior observations
    ei_am = 2.5  # Earth interior airmass limit
    ei_elong_req = 45.0  # Solar elongation required for inner solar system
    ei_area_req = 0.0  # Sky area required before attempting inner solar system
    per_night = True  # Dither DDF per night
    camera_ddf_rot_limit = 75.0  # degrees

    # Be sure to also update and regenerate DDF grid save file
    # if changing mjd_start
    mjd_start = SURVEY_START_MJD + mjd_plus

    fileroot, extra_info = set_run_info(dbroot=dbroot, file_end="v4.3.1_", out_dir=out_dir)

    pattern_dict = {
        1: [True],
        2: [True, False],
        3: [True, False, False],
        4: [True, False, False, False],
        # 4 on, 4 off
        5: [True, True, True, True, False, False, False, False],
        # 3 on 4 off
        6: [True, True, True, False, False, False, False],
        7: [True, True, False, False, False, False],
    }
    ei_night_pattern = pattern_dict[ei_night_pattern]
    reverse_ei_night_pattern = [not val for val in ei_night_pattern]

    sky = CurrentAreaMap(nside=nside)
    footprints_hp_array, labels = sky.return_maps()

    wfd_indx = np.where((labels == "lowdust") | (labels == "virgo"))[0]
    wfd_footprint = footprints_hp_array["r"] * 0
    wfd_footprint[wfd_indx] = 1

    footprints_hp = {}
    for key in footprints_hp_array.dtype.names:
        footprints_hp[key] = footprints_hp_array[key]

    footprint_mask = footprints_hp["r"] * 0
    footprint_mask[np.where(footprints_hp["r"] > 0)] = 1

    repeat_night_weight = None

    # Use the Almanac to find the position of the sun at the start of survey
    almanac = Almanac(mjd_start=mjd_start)
    sun_moon_info = almanac.get_sun_moon_positions(mjd_start)
    sun_ra_start = sun_moon_info["sun_RA"].copy()

    footprints = make_rolling_footprints(
        fp_hp=footprints_hp,
        mjd_start=mjd_start,
        sun_ra_start=sun_ra_start,
        nslice=nslice,
        scale=rolling_scale,
        nside=nside,
        wfd_indx=wfd_indx,
        order_roll=1,
        n_cycles=3,
        uniform=rolling_uniform,
    )

    gaps_night_pattern = [True] + [False] * nights_off

    long_gaps = gen_long_gaps_survey(
        nside=nside,
        footprints=footprints,
        night_pattern=gaps_night_pattern,
        u_exptime=u_exptime,
        nexp=nexp,
    )

    # Set up the DDF surveys to dither
    u_detailer = detailers.BandNexp(bandname="u", nexp=1, exptime=u_exptime)
    dither_detailer = detailers.DitherDetailer(per_night=per_night, max_dither=max_dither)
    details = [
        detailers.CameraRotDetailer(min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit),
        dither_detailer,
        u_detailer,
        detailers.Rottep2RotspDesiredDetailer(),
        detailers.LabelRegionsAndDDFs(),
    ]
    euclid_detailers = [
        detailers.CameraRotDetailer(min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit),
        detailers.EuclidDitherDetailer(),
        u_detailer,
        detailers.Rottep2RotspDesiredDetailer(),
        detailers.LabelRegionsAndDDFs(),
    ]
    ddfs = ddf_surveys(
        detailers=details,
        offseason_length=ddf_offseason_length,
        euclid_detailers=euclid_detailers,
        nside=nside,
        nexp=nexp,
    )

    greedy = gen_greedy_surveys(nside, nexp=nexp, footprints=footprints)
    neo = generate_twilight_near_sun(
        nside,
        night_pattern=ei_night_pattern,
        bands=ei_bands,
        n_repeat=ei_repeat,
        footprint_mask=footprint_mask,
        max_airmass=ei_am,
        max_elong=ei_elong_req,
        min_area=ei_area_req,
    )
    blobs = generate_blobs(
        nside,
        nexp=nexp,
        footprints=footprints,
        mjd_start=mjd_start,
        u_exptime=u_exptime,
    )
    twi_blobs = generate_twi_blobs(
        nside,
        nexp=nexp,
        footprints=footprints,
        wfd_footprint=wfd_footprint,
        repeat_night_weight=repeat_night_weight,
        night_pattern=reverse_ei_night_pattern,
    )

    roman_surveys = [
        gen_roman_on_season(nexp=nexp, exptime=29.2),
        gen_roman_off_season(nexp=nexp, exptime=29.2),
    ]

    if too:
        too_scale = 1.0
        sim_ToOs, event_table = gen_all_events(scale=too_scale, nside=nside)
        camera_rot_limits = [-80.0, 80.0]
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
        )
        detailer_list.append(detailers.LabelRegionsAndDDFs())
        # Let's make a footprint to follow up ToO events
        too_footprint = footprints_hp["r"] * 0 + np.nan
        too_footprint[np.where(footprints_hp["r"] > 0)[0]] = 1.0

        detailer_list.append(detailers.Rottep2RotspDesiredDetailer())
        toos = gen_too_surveys(
            nside=nside,
            detailer_list=detailer_list,
            too_footprint=too_footprint,
            split_long=split_long,
            n_snaps=nexp,
        )
        surveys = [toos, roman_surveys, ddfs, long_gaps, blobs, twi_blobs, neo, greedy]

    else:
        surveys = [roman_surveys, ddfs, long_gaps, blobs, twi_blobs, neo, greedy]

        sim_ToOs = None
        event_table = None
        fileroot = fileroot.replace("baseline", "no_too")

    scheduler = CoreScheduler(surveys, nside=nside)

    if args.setup_only:
        return scheduler
    else:
        years = np.round(survey_length / 365.25)
        observatory, scheduler, observations = run_sched(
            scheduler,
            survey_length=survey_length,
            verbose=verbose,
            filename=os.path.join(fileroot + "%iyrs.db" % years),
            extra_info=extra_info,
            nside=nside,
            illum_limit=illum_limit,
            mjd_start=mjd_start,
            event_table=event_table,
            sim_to_o=sim_ToOs,
            snapshot_dir=snapshot_dir,
        )
        return observatory, scheduler, observations


def sched_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print more output")
    parser.set_defaults(verbose=False)
    parser.add_argument("--survey_length", type=float, default=365.25 * 10, help="Survey length in days")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory")
    parser.add_argument("--nexp", type=int, default=2, help="Number of exposures per visit")
    parser.add_argument("--dbroot", type=str, help="Database root")
    parser.add_argument(
        "--setup_only",
        dest="setup_only",
        default=False,
        action="store_true",
        help="Only construct scheduler, do not simulate",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=DEFAULT_NSIDE,
        help="Nside should be set to default (32) except for tests.",
    )
    parser.add_argument(
        "--mjd_plus",
        type=float,
        default=0,
        help="number of days to add to the mjd start",
    )
    parser.add_argument(
        "--split_long",
        dest="split_long",
        action="store_true",
        help="Split long ToO exposures into standard visit lengths",
    )
    parser.add_argument("--snapshot_dir", type=str, default="", help="Directory for scheduler snapshots.")
    parser.set_defaults(split_long=False)
    parser.add_argument("--no_too", dest="no_too", action="store_true")
    parser.set_defaults(no_too=False)

    return parser


if __name__ == "__main__":
    parser = sched_argparser()
    args = parser.parse_args()
    gen_scheduler(args)
