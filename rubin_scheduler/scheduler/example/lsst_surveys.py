__all__ = (
    "safety_masks",
    "standard_bf",
    "gen_long_gaps_survey",
    "gen_template_surveys",
    "gen_greedy_surveys",
    "generate_blobs",
    "generate_short_blobs",
    "generate_twilight_near_sun",
)

import copy
from typing import Any

import numpy as np
import numpy.typing as npt

import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.surveys import BlobSurvey, GreedySurvey, LongGapSurvey, ScriptedSurvey
from rubin_scheduler.scheduler.utils import ConstantFootprint, Footprints, ecliptic_area
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD

# Set up values to use as kwarg defaults.
NEXP = 1
U_NEXP = 1
EXPTIME = 30.0
U_EXPTIME = 38.0
GOOD_SEEING_MAX = 1.3
CAMERA_ROT_LIMITS = (-80.0, 80.0)
SCIENCE_PROGRAM = "BLOCK-407"

BLOB_SURVEY_PARAMS_DEFAULTS = {
    "slew_approx": 7.5,
    "band_change_approx": 140.0,
    "read_approx": 3.07,
    "flush_time": 30.0,
    "smoothing_kernel": None,
    "nside": DEFAULT_NSIDE,
    "seed": 42,
    "dither": "night",
    "twilight_scale": True,
}


def safety_masks(
    nside: int = DEFAULT_NSIDE,
    moon_distance: float = 30,
    wind_speed_maximum: float = 20.0,
    min_alt: float = 20,
    max_alt: float = 86.5,
    min_az: float = 0,
    max_az: float = 360,
    shadow_minutes: float = 70,
    min_az_sunrise: float = 120,
    max_az_sunrise: float = 290,
    time_to_sunrise: float = 3.0,
    apply_time_limited_shadow: bool = True,
) -> list[bf.BaseBasisFunction]:
    """Basic safety mask basis functions.

    Avoids the moon, bright planets, high wind, and
    areas on the sky out of bounds, using
    the MoonAvoidanceBasisFunction, PlanetMaskBasisFunction,
    AvoidDirectWindBasisFunction, and the AltAzShadowMaskBasisFunction.
    Adds the default AltAzShadowMaskTimeLimited basis function to avoid
    pointing toward sunrise late in the night during commissioning.

    Parameters
    ----------
    nside : `int` or None
        The healpix nside to use.
        Default of None uses rubin_scheduler.utils.get_default_nside.
    moon_distance : `float`, optional
        Moon avoidance distance, in degrees.
    wind_speed_maximum : `float`, optional
        Wind speed maximum to apply to the wind avoidance basis function,
        in m/s.
    min_alt : `float`, optional
        Minimum altitude (in degrees) to observe.
    max_alt : `float`, optional
        Maximum altitude (in degrees) to observe.
    min_az : `float`, optional
        Minimum azimuth angle (in degrees) to observe.
    max_az : `float`, optional
        Maximum azimuth angle (in degrees) to observe.
    shadow_minutes : `float`, optional
        Avoid inaccessible alt/az regions, as well as parts of the sky
        which will move into those regions within `shadow_minutes` (minutes).
        Should be set to the expected time needed to execute the
        observing block associated with the survey.
    min_az_sunrise : `float`, optional
        Minimum azimuth angle (in degrees) to observe during time period
        at the end of the night (during time_to_sunrise).
    max_az_sunrise: `float`, optional
        Maximum azimuth angle (in degrees) to observe during the time period
        at the end of the night (during time_to_sunrise).
    time_to_sunrise : `float`, optional
        Hours before daybreak (sun @ alt=0) to start the azimuth avoidance
        mask.
    apply_time_limited_shadow : `float`, optional
        Flag for whether to apply the morning (time_to_sunrise) azimuth mask.

    Returns
    -------
    mask_basis_functions : `list` [`BaseBasisFunction`]
        Mask basis functions should always be used with a weight of 0.
        The masked (np.nan or -np.inf) regions will remain masked,
        but the basis function values won't influence the reward.
    """
    mask_bfs = []
    # Avoid the moon - too close to the moon will trip the REBs
    mask_bfs.append(bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance))
    # Avoid bright planets
    mask_bfs.append(bf.PlanetMaskBasisFunction(nside=nside))
    # Avoid the wind
    mask_bfs.append(bf.AvoidDirectWind(nside=nside, wind_speed_maximum=wind_speed_maximum))
    # Avoid the alt/az limits - this will pick up limits from the
    # yaml file configurations for the summit as well
    mask_bfs.append(
        bf.AltAzShadowMaskBasisFunction(
            nside=nside,
            min_alt=min_alt,
            max_alt=max_alt,
            min_az=min_az,
            max_az=max_az,
            shadow_minutes=shadow_minutes,
        )
    )

    if apply_time_limited_shadow:
        # Only look away from the azimuth of the sun in the next day
        # permitting emergency dome closure
        mask_bfs.append(
            bf.AltAzShadowTimeLimitedBasisFunction(
                nside=nside,
                min_alt=min_alt,
                max_alt=max_alt,
                min_az=min_az_sunrise,
                max_az=max_az_sunrise,
                shadow_minutes=shadow_minutes,
                # Time until/after sun_keys in hours
                time_to_sun=time_to_sunrise + shadow_minutes / 60.0,
                # 'sunrise' is 0 degree sunrise
                sun_keys=["sunrise"],
            )
        )
    return mask_bfs


def standard_bf(
    nside: int = DEFAULT_NSIDE,
    bandname: str = "g",
    bandname2: str | None = "i",
    m5_weight: float = 6.0,
    footprint_weight: float = 1.5,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 3.0,
    footprints: Footprints | None = None,
    fiducial_fwhm: float = 1.3,
    season: float = 365.25,
    season_start_hour: float = -4.0,
    season_end_hour: float = 2.0,
    strict: bool = True,
    seeing_fwhm_max: float | None = None,
) -> list[tuple[bf.BaseBasisFunction, float]]:
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
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
    footprint_weight : `float`
        The weight on the survey footprint basis function.
    slewtime_weight : `float`
        The weight on the slewtime basis function.
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
    footprints : `rubin_scheduler.scheduler.utils.Footprints` object
        The desired footprints object. Default of None will work, but is likely
        not desirable.
    fiducial_fwhm : `float`
        The fiducial FWHM for the M5Diff Basis function.
    n_obs_template : `dict`
        The number of observations to take every season in each band.
    season : `float`
        The length of season (i.e., how long before templates expire) (days).
        Default 365.25.
    season_start_hour : `float`
        Hour angle limits to use when gathering templates.
        Default -4 (hours)
    season_end_hour : `float`
       Hour angle limits to use when gathering templates.
       Default +2 (hours)
    strict : `bool`
        If False, use BandChangeBasisFunction which rewards visits in the
        same bandpass as currently in-use.
        If True, use a StrictBandBasisFunction which rewards visits in the
        same bandpass as currently in-use, but also rewards visits in
        different filters if the moon rose/set or twilight ended/started,
        or if there was a large gap in observing.
    seeing_fwhm_max : `float`
        Seeing limit to pass to the FootprintBasisFunction - visits
        with delivered image quality > seeing_fwhm_max will not be counted.

    Returns
    -------
    basis_functions_weights : `list`
        list of tuple pairs (basis function, weight) that is
        (rubin_scheduler.scheduler.BasisFunction object, float)

    """

    bfs = []

    if bandname2 is not None:
        bfs.append(
            (
                bf.M5DiffBasisFunction(bandname=bandname, nside=nside, fiducial_FWHMEff=fiducial_fwhm),
                m5_weight / 2.0,
            )
        )
        bfs.append(
            (
                bf.M5DiffBasisFunction(bandname=bandname2, nside=nside, fiducial_FWHMEff=fiducial_fwhm),
                m5_weight / 2.0,
            )
        )

    else:
        bfs.append(
            (
                bf.M5DiffBasisFunction(bandname=bandname, nside=nside, fiducial_FWHMEff=fiducial_fwhm),
                m5_weight,
            )
        )

    if bandname2 is not None:
        bfs.append(
            (
                bf.FootprintBasisFunction(
                    bandname=bandname,
                    footprint=footprints,
                    out_of_bounds_val=np.nan,
                    nside=nside,
                    seeing_fwhm_max=seeing_fwhm_max,
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
                    seeing_fwhm_max=seeing_fwhm_max,
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
                    seeing_fwhm_max=seeing_fwhm_max,
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

    bandnames = [fn for fn in [bandname, bandname2] if fn is not None]
    bfs.append((bf.BandLoadedBasisFunction(bandnames=bandnames), 0))

    return bfs


def gen_template_surveys(
    footprints: Footprints,
    nside: int = DEFAULT_NSIDE,
    seeing_fwhm_max: float = GOOD_SEEING_MAX,
    band1s: list[str] = ["u", "g", "r", "i", "z", "y"],
    dark_only: list[str] = ["u", "g"],
    ignore_obs: str | list[str] = ["DD", "twilight_near_sun"],
    camera_rot_limits: tuple[float, float] = CAMERA_ROT_LIMITS,
    exptime: float = EXPTIME,
    nexp: int = NEXP,
    u_exptime: float = U_EXPTIME,
    u_nexp: int = U_NEXP,
    n_obs_template: dict | None = None,
    pair_time: float = 33.0,
    area_required: float = 50.0,
    HA_min: float = 2.5,
    HA_max: float = 24 - 2.5,
    max_alt: float = 76.0,
    m5_weight: float = 6.0,
    footprint_weight: float = 1.5,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 3.0,
    science_program: str = SCIENCE_PROGRAM,
    blob_survey_params: dict | None = None,
    safety_mask_params: dict | None = None,
) -> list[BlobSurvey]:
    """Coherent area surveys (BlobSurvey with single visit) that
    are intended to acquire template visits in a convenient yet
    aggressive manner.

    Parameters
    ----------
    footprints : `rubin_scheduler.scheduler.utils.Footprints`
        The Footprints object for the Surveys.
    nside : `int`
        Nside for the surveys.
    seeing_fwhm_max : `float`
        The maximum (model delivered) seeing acceptable for the template
        surveys. This limit is set to attempt to match the seeing limits
        used when building difference image coadds.
    band1s : `list` [ `str` ]
        The bands in which to obtain templates, within the Footprints.
    dark_only : `list` [ `str` ]
        The bands to only attempt during dark-time.
    ignore_obs : `str` or `list` [ `str` ]
        Strings to match within scheduler_note to flag observations to ignore.
    camera_rot_limits : `list` [ `float`, `float` ]
        Camera rotator limits (in degrees) for the dither rotation detailer.
    exptime : `float`
        The exposure time for grizy visits.
    nexp : `int`
        The number of exposures per visit for grizy visits.
    u_exptime : `float`
        The exposure time for u band visits.
    u_nexp : `int`
        The number of exposures per visit for u band visits.
    n_obs_template : `dict` { `str` : `int` }
        Number of visits per bandpass before masking the Survey healpix.
        When this many visits are acquired, the pixel is considered "done".
    pair_time : `float`
        The time until the end of the first pass of the Blob.
        Since there is no second filter, this is the amount of time
        spent in the Blob.
    area_required : `float`
        The area required that needs templates, before the BlobSurvey will
        activate. Square degrees.
    HA_min : `float`
        The minimum HA to consider when considering template area.
    HA_max : `float`
        The maximum HA to consider when considering template area.
    max_alt : `float`
        The maximum altitude to use for the Surveys.
        Typically for BlobSurveys this is set lower than the max available,
        to about 76 degrees, to avoid long dome slews near azimuth.
        This is masked separately from the `safety_masks`.
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
    footprint_weight : `float`
        The weight on the survey footprint basis function.
    slewtime_weight : `float`
        The weight on the slewtime basis function.
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
    science_program : `str`
        The science_program to use for visits from these surveys.
    blob_survey_params : `dict` or None
        A dictionary of additional kwargs to pass to the BlobSurvey.
        In particular, the times for typical slews, readtime, etc. are
        useful for setting the number of pointings to schedule within
        pair_time.
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to mass to the standard safety masks.
    """

    if n_obs_template is None:
        n_obs_template = {"u": 4, "g": 4, "r": 4, "i": 4, "z": 4, "y": 4}

    if blob_survey_params is None:
        blob_survey_params = BLOB_SURVEY_PARAMS_DEFAULTS

    if safety_mask_params is None:
        safety_mask_params = {}
        safety_mask_params["nside"] = nside
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)
    if "shadow_minutes" not in safety_mask_params or safety_mask_params["shadow_minutes"] < pair_time:
        # Shadow minutes is only pair_time, only taking 1 image per field
        safety_mask_params["shadow_minutes"] = pair_time

    surveys = []

    for bandname in band1s:
        # Set up Detailers for camera rotator and ordering in altitude.
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
        )
        detailer_list.append(detailers.CloseAltDetailer())
        detailer_list.append(detailers.BandNexp(bandname="u", nexp=u_nexp, exptime=u_exptime))
        detailer_list.append(detailers.LabelRegionsAndDDFs())

        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        bfs.extend(
            standard_bf(
                nside,
                bandname=bandname,
                bandname2=None,
                footprints=footprints,
                m5_weight=m5_weight,
                footprint_weight=footprint_weight,
                slewtime_weight=slewtime_weight,
                stayband_weight=stayband_weight,
                seeing_fwhm_max=seeing_fwhm_max,
            )
        )

        bfs.append(
            (
                bf.AltAzShadowMaskBasisFunction(
                    nside=nside, shadow_minutes=pair_time, max_alt=max_alt, pad=3.0
                ),
                0,
            )
        )

        # not in twilight bf
        bfs.append((bf.TimeToTwilightBasisFunction(time_needed=pair_time), 0.0))
        bfs.append((bf.NotTwilightBasisFunction(), 0.0))
        bfs.append((bf.RevHaMaskBasisFunction(ha_min=HA_min, ha_max=HA_max, nside=nside), 0.0))

        # Need a once in night mask
        bfs.append((bf.NInNightMaskBasisFunction(n_limit=1, nside=nside), 0.0))

        # If u or g, only when moon is down
        if bandname in dark_only:
            bfs.append((bf.MoonAltLimitBasisFunction(alt_limit=-5), 0.0))

        # limit to first year
        bfs.append((bf.OnlyBeforeNightBasisFunction(night_max=366), 0.0))

        # Limit to only good seeing visits.
        bfs.append(
            (
                bf.MaskPoorSeeing(bandname, nside=nside, seeing_fwhm_max=seeing_fwhm_max),
                0,
            )
        )

        # Mask anything observed n_obs_template times reseting each season
        bfs.append(
            (
                bf.MaskAfterNObsSeeingBasisFunction(
                    nside=nside,
                    n_max=n_obs_template[bandname],
                    bandname=bandname,
                    seeing_fwhm_max=seeing_fwhm_max,
                ),
                0.0,
            )
        )

        # Add safety masks
        masks = safety_masks(**safety_mask_params)
        for m in masks:
            bfs.append((m, 0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]

        survey_name = "templates, %s" % bandname

        surveys.append(
            BlobSurvey(
                basis_functions,
                weights,
                bandname1=bandname,
                bandname2=None,
                exptime=exptime,
                ideal_pair_time=pair_time,
                survey_name=survey_name,
                science_program=science_program,
                observation_reason=f"template_blob_{bandname}_{pair_time:.1f}",
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                area_required=area_required,
                **blob_survey_params,
            )
        )

    return surveys


def blob_for_long(
    footprints: Footprints,
    nside: int = DEFAULT_NSIDE,
    band1s: list[str] = ["g"],
    band2s: list[str] = ["i"],
    ignore_obs: str | list[str] = ["DD", "twilight_near_sun", "ToO"],
    camera_rot_limits: tuple[float, float] = CAMERA_ROT_LIMITS,
    exptime: float = EXPTIME,
    nexp: int = NEXP,
    u_exptime: float = U_EXPTIME,
    u_nexp: int = U_NEXP,
    n_obs_template: dict | None = None,
    pair_time: float = 33.0,
    season: float = 365.25,
    season_start_hour: float = -4.0,
    season_end_hour: float = 2.0,
    HA_min: float = 12,
    HA_max: float = 24 - 3.5,
    max_alt: float = 76.0,
    m5_weight: float = 6.0,
    footprint_weight: float = 1.5,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 3.0,
    night_pattern: list[bool] = [True, True],
    time_after_twi: float = 30.0,
    blob_names: list[str] = [],
    scheduled_respect: float = 30.0,
    science_program: str = SCIENCE_PROGRAM,
    observation_reason: str | None = None,
    blob_survey_params: dict | None = None,
    safety_mask_params: dict | None = None,
    pair_pad: float = 5.0,
) -> list[BlobSurvey]:
    """
    Generate surveys that take observations in blobs.

    Parameters
    ----------
    footprints : `rubin_scheduler.scheduler.utils.Footprints`
        The Footprints object for the Surveys.
    nside : `int`
        The HEALpix nside to use. Default to DEFAULT_NSIDE.
    band1s : `list` [`str`]
        The bandnames for the first band in a pair.
        Default ["g"].
    band2s : `list` of `str`
        The band names for the second in the pair (None if unpaired).
        Default ["i"].
    ignore_obs : `str` or `list` of `str`
        Ignore observations by surveys that include the given substring(s).
    camera_rot_limits : `list` of `float`
        The limits to impose when rotationally dithering the camera (degrees).
        Default [-80., 80.].
    exptime : `float`
        The exposure time for grizy visits.
    nexp : `int`
        The number of exposures per visit for grizy visits.
    u_exptime : `float`
        The exposure time for u band visits.
    u_nexp : `int`
        The number of exposures per visit for u band visits.
    n_obs_template : `dict`
        The number of observations to take every season in each band.
        If None, sets to 3 each. Default None.
    pair_time : `float`
        The ideal time between pairs (minutes). Default 33.
    season : float
        The length of season (i.e., how long before templates expire) (days)
        Default 365.25.
    season_start_hour : `float`
        Hour angle limits to use when gathering templates.
        Default -4 (hours)
    sesason_end_hour : `float`
       Hour angle limits to use when gathering templates.
       Default +2 (hours)
    HA_min : `float`
        The minimum HA to consider when considering template area.
    HA_max : `float`
        The maximum HA to consider when considering template area.
    max_alt : `float`
        The maximium altitude to use when masking zenith (degrees).
        Typically for BlobSurveys this is set lower than the max available,
        to about 76 degrees, to avoid long dome slews near azimuth.
        This is masked separately from the `safety_masks`.
        Default 76.
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
    footprint_weight : `float`
        The weight on the survey footprint basis function.
    slewtime_weight : `float`
        The weight on the slewtime basis function.
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
    night_pattern : `list` [ `bool` ]
        Flips the blob_for_long surveys on or off, on a pattern of
        nights. These surveys typically don't execute every night.
    time_after_twi : `float`
        Don't start before this many minutes pass after -18 degree twilight.
    blob_names : `list` [ `str` ]
        Strings to match in scheduler_note to ensure surveys don't execute
        more than once per night.
    scheduled_respect : `float`
        Ensure that blobs don't start within this many minutes of scheduled
        observations (from a ScriptedSurvey).
    blob_survey_params : `dict` or None
        A dictionary of additional kwargs to pass to the BlobSurvey.
        In particular, the times for typical slews, readtime, etc. are
        useful for setting the number of pointings to schedule within
        pair_time.
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to mass to the standard safety masks.
    pair_pad : `float`
        How much extra time to pad above the necessary pair time
        for shadow basis function.
    """

    if blob_survey_params is None:
        blob_survey_params = BLOB_SURVEY_PARAMS_DEFAULTS
    if safety_mask_params is None:
        safety_mask_params = {"nside": nside}
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)

    shadow_minutes = pair_time * 2 + pair_pad
    if "shadow_minutes" not in safety_mask_params or safety_mask_params["shadow_minutes"] < shadow_minutes:
        safety_mask_params["shadow_minutes"] = shadow_minutes

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
        detailer_list.append(detailers.BandNexp(bandname="u", nexp=u_nexp, exptime=u_exptime))
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
                footprints=footprints,
                season=season,
                season_start_hour=season_start_hour,
                season_end_hour=season_end_hour,
            )
        )

        # Make sure we respect scheduled observations
        bfs.append((bf.TimeToScheduledBasisFunction(time_needed=scheduled_respect), 0))

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

        # Add safety masks
        masks = safety_masks(**safety_mask_params)
        for m in masks:
            bfs.append((m, 0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if bandname2 is None:
            survey_name = "blob_long, %s" % bandname
        else:
            survey_name = "blob_long, %s%s" % (bandname, bandname2)
        if bandname2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(bandname=bandname2))

        if observation_reason is None:
            observation_reason = f"triplet_pairs_{bandname}{bandname2}_{pair_time:.1f}"

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
                science_program=science_program,
                observation_reason=observation_reason,
                **blob_survey_params,
            )
        )

    return surveys


def gen_long_gaps_survey(
    footprints: Footprints,
    nside: int = DEFAULT_NSIDE,
    camera_rot_limits: tuple[float, float] = CAMERA_ROT_LIMITS,
    exptime: float = EXPTIME,
    nexp: int = NEXP,
    u_exptime: float = U_EXPTIME,
    u_nexp: int = U_NEXP,
    pair_time: float = 33.0,
    night_pattern: list[bool] = [True, True],
    gap_range: list[float] = [2, 7],
    HA_min: float = 12,
    HA_max: float = 24 - 3.5,
    time_after_twi: float = 120,
    m5_weight: float = 6.0,
    footprint_weight: float = 1.5,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 3.0,
    science_program: str = SCIENCE_PROGRAM,
    blob_survey_params: dict | None = None,
    safety_mask_params: dict | None = None,
    pair_pad: float = 5,
) -> list[LongGapSurvey]:
    """Generate long-gaps (triplets) surveys.

    Parameters
    -----------
    footprints : `rubin_scheduler.scheduler.utils.footprints.Footprints`
        The footprints to be used for the long-gaps surveys.
    nside : `int`
        The nside for the surveys.
    camera_rot_limits : `list` of `float`
        The limits to impose when rotationally dithering the camera (degrees).
        Default [-80., 80.].
    exptime : `float`
        The exposure time for grizy visits.
    nexp : `int`
        The number of exposures per visit for grizy visits.
    u_exptime : `float`
        The exposure time for u band visits.
    u_nexp : `int`
        The number of exposures per visit for u band visits.
    pair_time : `float`
        The ideal time between pairs (minutes). Default 33.
    night_pattern : `list` [ `bool` ]
        Which nights to let the survey execute.
        Default of [True, True] executes every night.
    gap_range : `list` [ `float` ]
        Range of times to attempt to gather pairs (hours).
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
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
    footprint_weight : `float`
        The weight on the survey footprint basis function.
    slewtime_weight : `float`
        The weight on the slewtime basis function.
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
    science_program : `str`
        The science_program to use for visits from these surveys.
    blob_survey_params : `dict` or None
        A dictionary of additional kwargs to pass to the BlobSurvey.
        In particular, the times for typical slews, readtime, etc. are
        useful for setting the number of pointings to schedule within
        pair_time.
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to mass to the standard safety masks.
    pair_pad : `float`
        How much extra time to pad above the necessary pair time
        for shadow basis function.
    """
    # Only copy the safety_mask_params here for the ScriptedSurvey.
    # The blob_for_long survey will copy/update on its own.
    safety_mask_params_scripted: dict[str, Any] = {"nside": nside}
    if safety_mask_params is not None:
        for key in safety_mask_params:
            safety_mask_params_scripted[key] = safety_mask_params[key]
            if key == "shadow_minutes":
                if safety_mask_params[key] < pair_time:
                    safety_mask_params_scripted[key] = pair_time
    if "shadow_minutes" not in safety_mask_params_scripted:
        safety_mask_params_scripted["shadow_minutes"] = pair_time

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
            camera_rot_limits=camera_rot_limits,
            exptime=exptime,
            nexp=nexp,
            u_exptime=u_exptime,
            u_nexp=u_nexp,
            pair_time=pair_time,
            nside=nside,
            band1s=[bandname1],
            band2s=[bandname2],
            night_pattern=night_pattern,
            time_after_twi=time_after_twi,
            HA_min=HA_min,
            HA_max=HA_max,
            m5_weight=m5_weight,
            footprint_weight=footprint_weight,
            slewtime_weight=slewtime_weight,
            stayband_weight=stayband_weight,
            blob_names=blob_names,
            science_program=science_program,
            blob_survey_params=blob_survey_params,
            safety_mask_params=safety_mask_params,
            pair_pad=pair_pad,
        )
        masks = safety_masks(**safety_mask_params_scripted)
        scripted = ScriptedSurvey(
            masks,
            nside=nside,
            ignore_obs=["blob", "DDF", "twi", "pair", "templates", "ToO"],
            science_program=science_program,
            detailers=[detailers.LabelRegionsAndDDFs()],
        )
        surveys.append(LongGapSurvey(blob[0], scripted, gap_range=gap_range, avoid_zenith=True))

    return surveys


def gen_greedy_surveys(
    nside: int = DEFAULT_NSIDE,
    bands: list[str] = ["r", "i", "z", "y"],
    ignore_obs: list[str] = ["DD", "twilight_near_sun", "ToO"],
    camera_rot_limits: tuple[float, float] = CAMERA_ROT_LIMITS,
    exptime: float = EXPTIME,
    nexp: int = NEXP,
    u_exptime: float = U_EXPTIME,
    u_nexp: int = U_NEXP,
    shadow_minutes: float = 15.0,
    max_alt: float = 76.0,
    m5_weight: float = 3.0,
    footprint_weight: float = 0.75,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 100.0,
    repeat_weight: float = -1.0,
    footprints: Footprints | None = None,
    science_program: str = SCIENCE_PROGRAM,
    safety_mask_params: dict | None = None,
) -> list[GreedySurvey]:
    """Generate greedy (single-best choice visits) Surveys.

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use
    bands : `list` [ `str` ]
        Bands in which to generate greedy surveys.
        Default ['r', 'i', 'z', 'y'].
    ignore_obs : `str` or `list` of `str`
        Ignore observations by surveys that include the given substring(s).
    camera_rot_limits : `list` [ `float` ]
        The limits to impose when rotationally dithering the camera (degrees).
        Default [-80., 80.].
    exptime : `float`
        The exposure time for grizy visits.
    nexp : `int`
        The number of exposures per visit for grizy visits.
    u_exptime : `float`
        The exposure time for u band visits.
    u_nexp : `int`
        The number of exposures per visit for u band visits.
    shadow_minutes : `float`
        Used to mask regions around zenith (minutes).
    max_alt : `float`
        The maximium altitude to use when masking zenith (degrees).
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
    footprint_weight : `float`
        The weight on the survey footprint basis function.
    slewtime_weight : `float`
        The weight on the slewtime basis function.
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
    repeat_weight : `float`
        Weight that enhances (+ values) or decreases (- values) the likelihood
        of revisiting the same pointing within a two-hour time gap.
    footprints : `rubin_scheduler.scheduler.utils.footprints.Footprints`
        The footprints to be used for the long-gaps surveys.
    science_program : `str`
        The science_program to use for visits from these surveys.
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to mass to the standard safety masks.
    """
    # Define the extra parameters that are used in the greedy survey. I
    # think these are fairly set, so no need to promote to utility func kwargs
    greed_survey_params = {
        "block_size": 1,
        "smoothing_kernel": None,
        "seed": 42,
        "camera": "LSST",
        "dither": "night",
    }
    if safety_mask_params is None:
        safety_mask_params = {"nside": nside}
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)
    if "shadow_minutes" not in safety_mask_params or safety_mask_params["shadow_minutes"] < shadow_minutes:
        safety_mask_params["shadow_minutes"] = shadow_minutes

    surveys = []
    detailer_list = [
        detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
    ]
    detailer_list.append(detailers.LabelRegionsAndDDFs())

    if "u" in bands:
        detailer_list.append(detailers.BandNexp(bandname="u", nexp=u_nexp, exptime=u_exptime))

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
                footprints=footprints,
                strict=False,
            )
        )

        # XXX-magic numbers
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

        masks = safety_masks(nside=nside, shadow_minutes=shadow_minutes)
        for m in masks:
            bfs.append((m, 0))

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
                survey_name=f"greedy {bandname}",
                science_program=science_program,
                observation_reason=f"singles_{bandname}",
                **greed_survey_params,
            )
        )

    return surveys


def generate_blobs(
    footprints: Footprints,
    nside: int = DEFAULT_NSIDE,
    band1s: list[str] = ["u", "u", "g", "r", "i", "z", "y"],
    band2s: list[str] = ["g", "r", "r", "i", "z", "y", "y"],
    ignore_obs: str | list[str] = ["DD", "twilight_near_sun", "ToO"],
    camera_rot_limits: tuple[float, float] = CAMERA_ROT_LIMITS,
    exptime: float = EXPTIME,
    nexp: int = NEXP,
    u_exptime: float = U_EXPTIME,
    u_nexp: int = U_NEXP,
    n_obs_template: dict | None = None,
    pair_time: float = 33.0,
    season: float = 365.25,
    season_start_hour: float = -4.0,
    season_end_hour: float = 2.0,
    max_alt: float = 76.0,
    seeing_fwhm_best: float = 0.8,
    m5_penalty_max: float = 0.5,
    m5_weight: float = 6.0,
    footprint_weight: float = 1.5,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 3.0,
    repeat_weight: float = -20,
    good_seeing: dict = {"g": 3, "r": 3, "i": 3},
    good_seeing_weight: float = 3.0,
    survey_start: float = SURVEY_START_MJD,
    scheduled_respect: float = 45.0,
    science_program: str = SCIENCE_PROGRAM,
    blob_survey_params: dict | None = None,
    safety_mask_params: dict | None = None,
    pair_pad: float = 5.0,
) -> list[BlobSurvey]:
    """Generate surveys that take observations in blobs.

    Parameters
    ----------
    footprints : `rubin_scheduler.scheduler.utils.Footprints`
        The Footprints object for the Surveys.
    nside : `int`
        The HEALpix nside to use. Default to DEFAULT_NSIDE.
    band1s : `list` [`str`]
        The bandnames for the first band in a pair.
    band2s : `list` of `str`
        The band names for the second in the pair (None if unpaired).
    ignore_obs : `str` or `list` of `str`
        Ignore observations by surveys that include the given substring(s).
    camera_rot_limits : `list` of `float`
        The limits to impose when rotationally dithering the camera (degrees).
    exptime : `float`
        The exposure time for grizy visits.
    nexp : `int`
        The number of exposures per visit for grizy visits.
    u_exptime : `float`
        The exposure time for u band visits.
    u_nexp : `int`
        The number of exposures per visit for u band visits.
    n_obs_template : `dict`
        The number of observations to take every season in each band.
        If None, sets to 3 each. Default None.
    pair_time : `float`
        The ideal time between pairs (minutes). Default 33.
    season : float
        The length of season (i.e., how long before templates expire) (days)
    season_start_hour : `float`
        Hour angle limits to use when gathering templates.
        Default -4 (hours)
    season_end_hour : `float`
       Hour angle limits to use when gathering templates.
       Default +2 (hours)
    max_alt : `float`
        The maximum altitude to use for the Surveys.
        Typically for BlobSurveys this is set lower than the max available,
        to about 76 degrees, to avoid long dome slews near azimuth.
        This is masked separately from the `safety_masks`.
    seeing_fwhm_best : `float`
        The FWHM effective to use when counting visits for the "GoodSeeing"
        basis function (not for templates; for deblending and shapes).
    m5_penalty_max : `float`
        The maximum penalty in 5-sigma limiting depth to consider
        still good for template building. Default 0.5 (mags).
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
    footprint_weight : `float`
        The weight on the survey footprint basis function.
    slewtime_weight : `float`
        The weight on the slewtime basis function.
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
    repeat_weight : `float`
        Weight that enhances (+ values) or decreases (- values) the likelihood
        of revisiting the same pointing within a two-hour time gap.
    good_seeing : `dict`
        Number of good-seeing (<6") images per band for basis function
        directing good-atmospheric seeing into particular bands.
    good_seeing_weight : `float`
        The weight to place on getting good-seeing images in the
        good_seeing bands.
    survey_start : `float`
        The mjd that the survey started (used for determining season for
        counting good seeing images within a season).
    scheduled_respect : `float`
        Ensure that blobs don't start within this many minutes of scheduled
        observations (from a ScriptedSurvey).
    science_program : `str`
        The science_program to use for visits from these surveys.
    blob_survey_params : `dict` or None
        A dictionary of additional kwargs to pass to the BlobSurvey.
        In particular, the times for typical slews, readtime, etc. are
        useful for setting the number of pointings to schedule within
        pair_time.
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to mass to the standard safety masks.
    pair_pad : `float`
        Padding to add to shadow mask. Default 5 (minutes).
    """

    if blob_survey_params is None:
        blob_survey_params = BLOB_SURVEY_PARAMS_DEFAULTS
    if safety_mask_params is None:
        safety_mask_params = {"nside": nside}
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)

    shadow_minutes = pair_time * 2 + pair_pad
    if "shadow_minutes" not in safety_mask_params or safety_mask_params["shadow_minutes"] < shadow_minutes:
        safety_mask_params["shadow_minutes"] = shadow_minutes
    if "max_alt" in safety_mask_params and safety_mask_params["max_alt"] > max_alt:
        safety_mask_params["max_alt"] = max_alt

    if n_obs_template is None:
        n_obs_template = {"u": 3, "g": 3, "r": 3, "i": 3, "z": 3, "y": 3}

    surveys = []

    times_needed = [pair_time, pair_time * 2]
    for bandname, bandname2 in zip(band1s, band2s):
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
        )
        detailer_list.append(detailers.CloseAltDetailer())
        if (bandname == "u") | (bandname2 == "u"):
            detailer_list.append(detailers.BandNexp(bandname="u", nexp=U_NEXP, exptime=u_exptime))
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
                footprints=footprints,
                season=season,
                season_start_hour=season_start_hour,
                season_end_hour=season_end_hour,
            )
        )

        # Suppress revisits within 3 hours of the first pair.
        # Without this, we tend to repeat fields too quickly
        # but repeats after 3 hours could be useful, so let it expire then.
        bfs.append(
            (
                bf.VisitRepeatBasisFunction(
                    gap_min=0, gap_max=3 * 60.0, bandname=None, nside=nside, npairs=20
                ),
                repeat_weight,
            )
        )

        # Insert things for getting good seeing images
        # Probably for galaxy shape measurements.
        if bandname2 is not None:
            if bandname in list(good_seeing.keys()):
                bfs.append(
                    (
                        bf.NGoodSeeingBasisFunction(
                            bandname=bandname,
                            nside=nside,
                            mjd_start=survey_start,
                            footprint=footprints.get_footprint(bandname),
                            n_obs_desired=good_seeing[bandname],
                            seeing_fwhm_max=seeing_fwhm_best,
                            m5_penalty_max=m5_penalty_max,
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
                            mjd_start=survey_start,
                            footprint=footprints.get_footprint(bandname2),
                            n_obs_desired=good_seeing[bandname2],
                            seeing_fwhm_max=seeing_fwhm_best,
                            m5_penalty_max=m5_penalty_max,
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
                            mjd_start=survey_start,
                            footprint=footprints.get_footprint(bandname),
                            n_obs_desired=good_seeing[bandname],
                            seeing_fwhm_max=seeing_fwhm_best,
                            m5_penalty_max=m5_penalty_max,
                        ),
                        good_seeing_weight,
                    )
                )
        # Make sure we respect scheduled observations
        bfs.append((bf.TimeToScheduledBasisFunction(time_needed=scheduled_respect), 0))

        if bandname2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.TimeToTwilightBasisFunction(time_needed=time_needed), 0.0))
        bfs.append((bf.NotTwilightBasisFunction(), 0.0))

        # Add safety masks
        masks = safety_masks(**safety_mask_params)
        for m in masks:
            bfs.append((m, 0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]

        # Set survey name
        if bandname2 is None:
            survey_name = "pair_%i, %s" % (pair_time, bandname)
        else:
            survey_name = "pair_%i, %s%s" % (pair_time, bandname, bandname2)
        if bandname2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(bandname=bandname2))

        observation_reason = f"pairs_{bandname}"
        if bandname2 is not None:
            observation_reason += f"{bandname2}"
        observation_reason += f"_{pair_time:.1f}"

        surveys.append(
            BlobSurvey(
                basis_functions,
                weights,
                bandname1=bandname,
                bandname2=bandname2,
                exptime=exptime,
                ideal_pair_time=pair_time,
                survey_name=survey_name,
                science_program=science_program,
                observation_reason=observation_reason,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                **blob_survey_params,
            )
        )

    return surveys


def generate_short_blobs(
    footprints: Footprints,
    nside: int = DEFAULT_NSIDE,
    band1s: list[str] = ["r", "i", "z", "y"],
    band2s: list[str] = ["i", "z", "y", "y"],
    ignore_obs: str | list[str] = ["DD", "twilight_near_sun", "ToO"],
    camera_rot_limits: tuple[float, float] = CAMERA_ROT_LIMITS,
    exptime: float = EXPTIME,
    nexp: int = NEXP,
    n_obs_template: dict[str, int] | None = None,
    pair_time: float = 15.0,
    season: float = 365.25,
    season_start_hour: float = -4.0,
    season_end_hour: float = 2.0,
    max_alt: float = 76.0,
    m5_weight: float = 6.0,
    footprint_weight: float = 1.5,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 3.0,
    repeat_weight: float = 0,
    scheduled_respect: float = 15.0,
    night_pattern: list[bool] | None = None,
    science_program: str = SCIENCE_PROGRAM,
    blob_survey_params: dict | None = None,
    safety_mask_params: dict | None = None,
    pair_pad: float = 5.0,
) -> list[BlobSurvey]:
    """
    Generate surveys that take observations in blobs, for twilight time.
    Shorter blobs, different weights for the basis functions.

    Parameters
    ----------
    footprints : `rubin_scheduler.scheduler.utils.Footprints`
        The Footprints object for the Surveys.
    nside : `int`
        The HEALpix nside to use. Default to DEFAULT_NSIDE.
    band1s : `list` [`str`]
        The bandnames for the first band in a pair.
    band2s : `list` [ `str` ]
        The band names for the second in the pair (None if unpaired).
    ignore_obs : `str` or `list` [ `str` ]
        Ignore observations by surveys that include the given substring(s).
    camera_rot_limits : `list` [ `float` ]
        The limits to impose when rotationally dithering the camera (degrees).
    exptime : `float`
        Exposure time for visits.
    nexp : `int`
        Number of exposures per visit.
    n_obs_template : `dict` or None
        The number of observations to take every season in each band.
        If None, sets to 3 each. Default None.
    pair_time : `float`
        The ideal time between pairs (minutes). Default 33.
    season : float
        The length of season (i.e., how long before templates expire) (days)
    season_start_hour : `float`
        Hour angle limits to use when gathering templates.
        Default -4 (hours)
    season_end_hour : `float`
       Hour angle limits to use when gathering templates.
       Default +2 (hours)
    max_alt : `float`
        The maximum altitude to use for the Surveys.
        Typically for BlobSurveys this is set lower than the max available,
        to about 76 degrees, to avoid long dome slews near azimuth.
        This is masked separately from the `safety_masks`.
    m5_weight : `float`
        The weight for the 5-sigma depth difference basis function.
    footprint_weight : `float`
        The weight on the survey footprint basis function.
    slewtime_weight : `float`
        The weight on the slewtime basis function.
    stayband_weight : `float`
        The weight on basis function that tries to stay avoid band changes.
    template_weight : `float`
        The weight to place on getting image templates every season.
    repeat_weight : `float`
        Weight that enhances (+ values) or decreases (- values) the likelihood
        of revisiting the same pointing within a two-hour time gap.
        Default of 0 here lets this survey repeat visits but total time
        in operation is limited anyway.
    scheduled_respect : `float`
        Ensure that blobs don't start within this many minutes of scheduled
        observations (from a ScriptedSurvey).
    night_pattern : `list` [ `bool` ]
        Which nights to let the survey execute (should be the opposite of
        the pattern for the NEO twilight survey).
        Default of [True, True] executes every night.
    science_program : `str`
        The science_program to use for visits from these surveys.
    blob_survey_params : `dict` or None
        A dictionary of additional kwargs to pass to the BlobSurvey.
        In particular, the times for typical slews, readtime, etc. are
        useful for setting the number of pointings to schedule within
        pair_time.
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to mass to the standard safety masks.
    pair_pad : `float`
        The padding to use for shadow mask. Default 5 (minutes).
    """

    if blob_survey_params is None:
        blob_survey_params = BLOB_SURVEY_PARAMS_DEFAULTS
    if safety_mask_params is None:
        safety_mask_params = {"nside": nside}
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)

    shadow_minutes = pair_time * 2 + pair_pad
    if shadow_minutes not in safety_mask_params or safety_mask_params["shadow_minutes"] < shadow_minutes:
        safety_mask_params["shadow_minutes"] = shadow_minutes

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
                footprints=footprints,
                season=season,
                season_start_hour=season_start_hour,
                season_end_hour=season_end_hour,
            )
        )

        # XXX--magic numbers. Kinda nebulous why here.
        bfs.append(
            (
                bf.VisitRepeatBasisFunction(
                    gap_min=0, gap_max=2 * 60.0, bandname=None, nside=nside, npairs=20
                ),
                repeat_weight,
            )
        )

        # Make sure we respect scheduled observations
        bfs.append((bf.TimeToScheduledBasisFunction(time_needed=scheduled_respect), 0))

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

        # Add safety masks
        masks = safety_masks(**safety_mask_params)
        for m in masks:
            bfs.append((m, 0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]

        # Set survey_name and observation_reason
        if bandname2 is None:
            survey_name = "pair_%i, %s" % (pair_time, bandname)
        else:
            survey_name = "pair_%i, %s%s" % (pair_time, bandname, bandname2)
        if bandname2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(bandname=bandname2))

        observation_reason = f"pairs_{bandname}"
        if bandname2 is not None:
            observation_reason += f"{bandname2}"
        observation_reason += f"_{pair_time:.1f}"

        surveys.append(
            BlobSurvey(
                basis_functions,
                weights,
                bandname1=bandname,
                bandname2=bandname2,
                exptime=exptime,
                ideal_pair_time=pair_time,
                survey_name=survey_name,
                science_program=science_program,
                observation_reason=observation_reason,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                **blob_survey_params,
            )
        )

    return surveys


def generate_twilight_near_sun(
    nside: int = DEFAULT_NSIDE,
    night_pattern: list[bool] | None = None,
    nexp: int = 1,
    exptime: float = 15,
    ideal_pair_time: float = 5.0,
    max_airmass: float = 2.0,
    camera_rot_limits: tuple[float, float] = CAMERA_ROT_LIMITS,
    time_needed: float = 10.0,
    footprint_mask: npt.NDArray | float = 1,
    footprint_weight: float = 0.1,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 3.0,
    band_dist_weight: float = 0.3,
    min_area: float | None = None,
    bands: str = "riz",
    n_repeat: int = 4,
    sun_alt_limit: float = -14.8,
    time_to_12deg: float = 25.0,
    slew_estimate: float = 4.5,
    max_elong: float = 60.0,
    ignore_obs: list[str] = ["DD", "pair", "long", "blob", "greedy", "template", "ToO"],
    science_program: str = SCIENCE_PROGRAM,
    safety_mask_params: dict | None = None,
) -> list[BlobSurvey]:
    """Generate a survey for observing NEO objects in twilight.

    This aims to cover ecliptic area within the LSST footprint, near the sun.

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use. Default to DEFAULT_NSIDE.
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
    footprint_mask : `np.ndarray` or `float`
        Mask to apply to the constructed ecliptic target mask (None).
        Keeps ecliptic area within this region.
        Default None
    footprint_weight : `float`
        Weight for footprint basis function. Default 0.1 (unitless).
    slewtime_weight : `float`
        Weight for slewtime basis function. Default 3 (unitless)
    stayband_weight : `float`
        Weight for staying in the same band basis function.
        Default 3 (unitless)
    band_dist_weight : `float`
        Weight for the basis function that tries to keep the distribution
        of visits across filters even.
    min_area : `float`
        The area that needs to be available before the survey will return
        observations (sq degrees). Default None.
    bands : `str`
        The bands to use, default 'riz'
    n_repeat : `int`
        The number of times a blob should be repeated, default 4.
    sun_alt_limit : `float`
        Do not start unless sun is higher than this limit (degrees).
        Default -14.8.
    time_to_sunrise : `float`
        Do not execute if time to sunrise is greater than (minutes).
        Default 25.
    slew_estimate : `float`
        An estimate of how long it takes to slew between
        neighboring fields (seconds). Default 4.5
    max_elong : `float`
        Maximum solar elongation to mini-allow survey to reach.
    ignore_obs : `str` or `list` [ `str` ]
        Ignore observations by surveys that include the given substring(s).
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to mass to the standard safety masks.
    """
    if safety_mask_params is None:
        safety_mask_params = {"nside": nside}
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)
    if (
        "shadow_minutes" not in safety_mask_params
        or safety_mask_params["shadow_minutes"] < ideal_pair_time * 4
    ):
        # pair_time * 4 (for the quad)
        safety_mask_params["shadow_minutes"] = ideal_pair_time * 4

    survey_name = "twilight_near_sun"
    observation_reason = "twilight_near_sun"
    footprint = ecliptic_area(nside=nside, mask=footprint_mask)
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

        bfs.append((bf.BandLoadedBasisFunction(bandnames=bandname), 0))
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
                bf.CloseToTwilightBasisFunction(
                    max_sun_alt_limit=sun_alt_limit, max_time_to_12deg=time_to_12deg
                ),
                0,
            )
        )

        # Add safety masks
        masks = safety_masks(**safety_mask_params)
        for m in masks:
            bfs.append((m, 0))

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
                dither="night",
                nexp=nexp,
                detailers=detailer_list,
                twilight_scale=False,
                area_required=min_area,
                observation_reason=observation_reason,
                science_program=science_program,
            )
        )
    return surveys
