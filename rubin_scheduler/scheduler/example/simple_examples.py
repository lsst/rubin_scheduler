__all__ = (
    "get_ideal_model_observatory",
    "update_model_observatory_sunset",
    "standard_masks",
    "simple_rewards",
    "simple_pairs_survey",
    "simple_greedy_survey",
    "simple_rewards_field_survey",
    "simple_field_survey",
)


import numpy as np
from astropy.time import Time

import rubin_scheduler.scheduler.basis_functions as basis_functions
import rubin_scheduler.scheduler.detailers as detailers
import rubin_scheduler.scheduler.features as features
from rubin_scheduler.scheduler.model_observatory import (
    KinemModel,
    ModelObservatory,
    rotator_movement,
    tma_movement,
)
from rubin_scheduler.scheduler.schedulers import BandSwapScheduler
from rubin_scheduler.scheduler.surveys import BlobSurvey, FieldSurvey, GreedySurvey
from rubin_scheduler.scheduler.utils import Footprint, get_current_footprint
from rubin_scheduler.site_models import Almanac, ConstantSeeingData, ConstantWindData
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD


def get_ideal_model_observatory(
    nside: int = DEFAULT_NSIDE,
    dayobs: str = "2024-09-09",
    fwhm_500: float = 1.6,
    wind_speed: float = 5.0,
    wind_direction: float = 340,
    tma_percent: float = 70,
    rotator_percent: float = 100,
    survey_start: Time = Time("2024-09-09T12:00:00", format="isot", scale="utc").mjd,
) -> ModelObservatory:
    """Set up a model observatory with constant seeing and wind speed,
    running on 'dayobs'.

    Track a constant 'survey_start' over different 'dayobs' so that the
    'night' value in the simulation outputs is consistent, and any
    basis functions which might use 'season' values from the conditions
    have appropriate values.

    Parameters
    ----------
    nside : `int`
        The nside for the model observatory.
    dayobs : `str`
        DAYOBS formatted str (YYYY-MM-DD) for the evening to start
        up the observatory.
    fwhm_500 : `float`, optional
        The value to set for atmospheric component of seeing,
         constant seeing throughout the night (arcseconds).
        Ad-hoc value for start of comcam on-sky operations about 2.0".
    wind_speed : `float`, optional
        Set a (constant) wind speed for the night, (m/s).
        Default of 5.0 is minimal but noticeable.
    wind_direction : `float`, optional
        Set a (constant) wind direction for the night (deg).
        Default of 340 is midrange between typical wind directions for
        the site (see SITCOMTN-126).
    tma_percent : `float`, optional
        Set a percent of full-performance for the telescope TMA (0-100).
        Value of 10(%) is likely for start of comcam on-sky SV surveys.
    rotator_percent : `float`, optional
        Set a percent of full-performance for the rotator.
        Default of 100% is likely for the start of comcam on-sky SV surveys.
    survey_start : `float`, optional
        MJD of the day of the survey start of operations.
        This should be kept constant for a given set of simulations,
        so that the "night" value in the output is consistent.
        For surveys which use basis functions depending on season, this
        is also important to be constant.

    Returns
    -------
    observatory : `~.scheduler.model_observatory.ModelObservatory`
        A ModelObservatory set up to start operations in the evening
        of DAYOBS.

    Notes
    -----
    The time for the model observatory will be advanced to the time
    of `sunset_start_key` (default -12 degree sunset) in the model
    observatory. The bands may not be correct however; use
    `update_model_observatory_sunset` to get bands in place.
    """
    # Set up a fresh model observatory
    mjd_now = Time(f"{dayobs}T12:00:00", format="isot", scale="utc").mjd

    kinematic_model = KinemModel(mjd0=mjd_now)
    rot = rotator_movement(rotator_percent)
    kinematic_model.setup_camera(**rot)
    tma = tma_movement(tma_percent)
    kinematic_model.setup_telescope(**tma)

    # Some weather telemetry that might be useful
    seeing_data = ConstantSeeingData(fwhm_500=fwhm_500)
    wind_data = ConstantWindData(wind_direction=wind_direction, wind_speed=wind_speed)

    # Set up the model observatory
    observatory = ModelObservatory(
        nside=nside,
        mjd=mjd_now,
        mjd_start=survey_start,
        kinem_model=kinematic_model,  # Modified kinematics
        cloud_data="ideal",  # No clouds
        seeing_data=seeing_data,  # Modified seeing
        wind_data=wind_data,  # Add some wind
        downtimes="ideal",  # No downtime
        lax_dome=True,  # dome crawl?
        init_load_length=1,  # size of skybrightness files to load first
    )
    return observatory


def update_model_observatory_sunset(
    observatory: ModelObservatory, band_scheduler: BandSwapScheduler, twilight: int | float = -12
) -> ModelObservatory:
    """Ensure correct bands are in place according to the band_scheduler.

    Parameters
    ----------
    observatory : `~.scheduler.model_observatory.ModelObservatory`
        The ModelObservatory simulating the observatory.
    band_scheduler : `~.scheduler.schedulers.BandScheduler`
        The band scheduler providing appropriate information on
        the bands that should be in place on the current observatory day.
    twilight : `int` or `float`
        If twilight is -12 or -18, the Almanac -12 or -18 degree twilight
        times are used to set the current observatory time.
        If  any other value is provided, it is assumed to be a specific
        MJD to start operating the observatory.
        Band choices are based on the time after advancing to twilight.

    Returns
    -------
    observatory :  `~.scheduler.model_observatory.ModelObservatory`
        The ModelObservatory simulating the observatory, updated to the
        time of 'twilight' and with mounted_bands matching
        the bands chosen by the band_scheduler for the current time
        at 'twilight'.
    """
    # Move to *next* possible sunset
    if twilight in (-12, -18):
        time_key = f"sun_n{twilight*-1 :0d}_setting"
        # Use the observatory almanac to find the exact time to forward to
        alm_indx = np.searchsorted(observatory.almanac.sunsets[time_key], observatory.mjd, side="right") - 1
        new_mjd = observatory.almanac.sunsets[time_key][alm_indx]
        if new_mjd < observatory.mjd:
            alm_indx += 1
            new_mjd = observatory.almanac.sunsets[time_key][alm_indx]
        observatory.mjd = new_mjd
    else:
        # Assume twilight was a particular (MJD) time
        observatory.mjd = twilight

    # Make sure correct bands are mounted
    conditions = observatory.return_conditions()
    bands_needed = band_scheduler(conditions)
    observatory.observatory.mount_bands(bands_needed)
    return observatory


def standard_masks(
    nside: int,
    moon_distance: float = 30.0,
    wind_speed_maximum: float = 20.0,
    min_alt: float = 20,
    max_alt: float = 86.5,
    min_az: float = 0,
    max_az: float = 360,
    shadow_minutes: float = 30,
) -> list[basis_functions.BaseBasisFunction]:
    """A set of standard mask functions.

    Avoids the moon, bright planets, high wind, and
    areas on the sky out of bounds, using
    the MoonAvoidanceBasisFunction, PlanetMaskBasisFunction,
    AvoidDirectWindBasisFunction, and the AltAzShadowMaskBasisFunction.

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

    Returns
    -------
    mask_basis_functions : `list` [`BaseBasisFunction`]
        Mask basis functions should always be used with a weight of 0.
        The masked (np.nan or -np.inf) regions will remain masked,
        but the basis function values won't influence the reward.
    """
    masks = []
    # Add the Moon avoidance mask
    masks.append(basis_functions.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance))
    # Add a mask around bright planets
    masks.append(basis_functions.PlanetMaskBasisFunction(nside=nside))
    # Add the wind avoidance mask
    masks.append(basis_functions.AvoidDirectWind(nside=nside, wind_speed_maximum=wind_speed_maximum))
    # Avoid inaccessible parts of the sky, as well as places that will
    # move into those places within shadow_minutes.
    masks.append(
        basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside,
            min_alt=min_alt,
            max_alt=max_alt,
            min_az=min_az,
            max_az=max_az,
            shadow_minutes=shadow_minutes,
        )
    )
    return masks


def simple_rewards(
    footprints: Footprint,
    bandname: str,
    nside: int = DEFAULT_NSIDE,
    m5_weight: float = 6.0,
    footprint_weight: float = 1.5,
    slewtime_weight: float = 3.0,
    stayband_weight: float = 3.0,
    repeat_weight: float = -20,
) -> list[basis_functions.BaseBasisFunction]:
    """A simple set of rewards for area-based surveys.

    Parameters
    ----------
    footprints : `Footprint`
        A Footprint class, which takes a target map and adds a
        time-dependent weighting on top (such as seasonal weighting).
    bandname : `str`
        The bandname active for these rewards.
    nside : `int`, optional
        The nside for the rewards.
    m5_weight : `float`, optional
        The weight to give the M5Diff basis function.
    footprint_weight : `float`, optional
        The weight to give the footprint basis function.
    slewtime_weight : `float`, optional
        The weight to give the slewtime basis function.
    stayband_weight : `float`, optional
        The weight to give the BandChange basis function.
    repeat_weight : `float`, optional
        The weight to give the VisitRepeat basis function.
        This is negative by default, to avoid revisiting the same part
        of the sky within `gap_max` (3 hours) in any band (except in the
        pairs survey itself).

    Returns
    -------
    reward_functions : `list` [(`BaseBasisFunction`, `float`)]
        List of tuples, each tuple is a reward function followed by
        its respective weight.

    Notes
    -----
    Increasing the m5_weight moves visits toward darker skies.
    Increasing the footprint weight distributes visits more evenly.
    Increasing the slewtime weight acquires visits with shorter slewtime.
    The balance in the defaults here has worked reasonably for pair
    surveys in simulations.
    """
    reward_functions = []
    # Add M5 basis function (rewards dark sky)
    reward_functions.append((basis_functions.M5DiffBasisFunction(bandname=bandname, nside=nside), m5_weight))
    # Add a footprint basis function
    # (rewards sky with fewer visits)
    reward_functions.append(
        (
            basis_functions.FootprintBasisFunction(
                bandname=bandname,
                footprint=footprints,
                out_of_bounds_val=np.nan,
                nside=nside,
            ),
            footprint_weight,
        )
    )
    # Add a reward function for small slewtimes.
    reward_functions.append(
        (
            basis_functions.SlewtimeBasisFunction(bandname=bandname, nside=nside),
            slewtime_weight,
        )
    )
    # Add a reward to stay in the same band as much as possible.
    reward_functions.append((basis_functions.BandChangeBasisFunction(bandname=bandname), stayband_weight))
    # And this is technically a mask, to avoid asking for observations
    # which are not possible. However, it depends on the bands
    # requested, compared to the bands available in the camera.
    reward_functions.append((basis_functions.BandLoadedBasisFunction(bandnames=bandname), 0))
    # And add a basis function to avoid repeating the same pointing
    # (VisitRepeat can be used to either encourage or discourage repeats,
    # depending on the repeat_weight value).
    reward_functions.append(
        (
            basis_functions.VisitRepeatBasisFunction(
                gap_min=0, gap_max=3 * 60.0, bandname=None, nside=nside, npairs=20
            ),
            repeat_weight,
        )
    )
    return reward_functions


def simple_pairs_survey(
    nside: int = DEFAULT_NSIDE,
    bandname: str = "g",
    bandname2: str | None = None,
    mask_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions_weights: list[float] | None = None,
    survey_start: float = SURVEY_START_MJD,
    footprints_hp: np.ndarray | None = None,
    footprint: Footprint | None = None,
    camera_rot_limits: list[float] = [-80.0, 80.0],
    pair_time: float = 30.0,
    exptime: float = 30.0,
    nexp: int = 1,
    science_program: str | None = None,
    observation_reason: str | None = None,
    dither: str = "night",
    camera_dither: str = "night",
    require_time: bool = False,
) -> BlobSurvey:
    """Set up a simple blob survey to acquire pairs of visits.

    Parameters
    ----------
    nside  : `int`, optional
        Nside for the surveys.
    bandname : `str`, optional
        Bandname for the first visit of the pair.
    bandname2 : `str` or None, optional
        Bandname for the second visit of the pair. If None, the
        first band will be used for both visits.
    mask_basis_functions : `list` [`BaseBasisFunction`] or None
        List of basis functions to use as masks (with implied weight 0).
        If None, `standard_masks` is used with default parameters.
    reward_basis_functions : `list` [`BaseBasisFunction`] or None
        List of basis functions to use as rewards.
        If None, a basic set of basis functions will be used.
    reward_basis_functions_weights : `list` [`float`] or None
        List of values to use as weights for the reward basis functions.
        If None, default values for the basic set will be used.
    survey_start : `float` or None
        The start of the survey, in MJD.
        If None, `survey_start_mjd()` is used.
        This should be the start of the survey, not the current time.
    footprints_hp : `np.ndarray` (N,) or None
        An array of healpix maps with the target survey area, with dtype
        like [(bandname, '<f8'), (bandname2, '<f8')].
        If None, `get_current_footprint()` will be used, which will cover
        the expected LSST survey footprint.
    camera_rot_limits : `list` [`float`]
        The rotator limits to expect for the camera.
        These should be slightly padded from true limits, to allow for
        slight delays between requesting observations and acquiring them.
    pair_time : `float`
        The ideal time between pairs of visits, in minutes.
    exptime : `float`
        The on-sky exposure time per visit.
    nexp : `int`
        The number of exposures per visit (exptime * nexp = total on-sky time).
    science_program : `str` | None
        The science_program for the Survey.
        This maps to the BLOCK and `science_program` in the consDB.
    observation_reason : `str` | None
        The observation_reason for the Survey.
        Indicates survey mode, potentially enhanced with particular reason.
    require_time : `bool`
        If True, add a mask basis function that checks there is enough
        time before twilight to execute the pairs. Default False.

    Returns
    -------
    pair_survey : `BlobSurvey`
        A blob survey configured to take pairs at spacing of pair_time,
        in bandname + bandname2.
    """

    # Use the Almanac to find the position of the sun at the start of survey.
    almanac = Almanac(mjd_start=survey_start)
    sun_moon_info = almanac.get_sun_moon_positions(survey_start)
    sun_ra_start = sun_moon_info["sun_RA"].copy()

    if footprint is None:
        if footprints_hp is None:
            footprints_hp, labels = get_current_footprint(nside=nside)
        footprints = Footprint(mjd_start=survey_start, sun_ra_start=sun_ra_start, nside=nside)
        for f in footprints_hp.dtype.names:
            footprints.set_footprint(f, footprints_hp[f])
    else:
        footprints = footprint

    if mask_basis_functions is None:
        mask_basis_functions = standard_masks(nside=nside)

    # Don't start a blob if there isn't time to finish it before twilight
    if require_time:
        time_needed = pair_time
        if bandname2 is not None:
            time_needed *= 2
        mask_basis_functions.append(basis_functions.TimeToTwilightBasisFunction(time_needed=time_needed))

    # Mask basis functions have zero weights.
    mask_basis_functions_weights = [0 for mask in mask_basis_functions]

    if reward_basis_functions is None:
        # Create list of (tuples of) basic reward basis functions and weights.
        m5_weight = 6.0
        footprint_weight = 1.5
        slewtime_weight = 3.0
        stayband_weight = 3.0
        repeat_weight = -20

        if bandname2 is None:
            reward_functions = simple_rewards(
                footprints=footprints,
                bandname=bandname,
                nside=nside,
                m5_weight=m5_weight,
                footprint_weight=footprint_weight,
                slewtime_weight=slewtime_weight,
                stayband_weight=stayband_weight,
                repeat_weight=repeat_weight,
            )

        else:
            # Add the same basis functions, but M5 and footprint
            # basis functions need to be added twice, with half the weight.
            rf1 = simple_rewards(
                footprints=footprints,
                bandname=bandname,
                nside=nside,
                m5_weight=m5_weight / 2.0,
                footprint_weight=footprint_weight / 2.0,
                slewtime_weight=slewtime_weight,
                stayband_weight=stayband_weight,
                repeat_weight=repeat_weight,
            )
            rf2 = simple_rewards(
                footprints=footprints,
                bandname=bandname2,
                nside=nside,
                m5_weight=m5_weight / 2.0,
                footprint_weight=footprint_weight / 2.0,
                slewtime_weight=0,
                stayband_weight=0,
                repeat_weight=0,
            )
            # Now clean up and combine these - and remove the separate
            # BasisFunction for BandLoadedBasisFunction.
            reward_functions = [(i[0], i[1]) for i in rf1 if i[1] > 0] + [
                (i[0], i[1]) for i in rf2 if i[1] > 0
            ]
            # Then put back in the BandLoadedBasisFunction with both bands.
            bandnames = [fn for fn in [bandname, bandname2] if fn is not None]
            reward_functions.append((basis_functions.BandLoadedBasisFunction(bandnames=bandnames), 0))

        # unpack the basis functions and weights
        reward_basis_functions_weights = [val[1] for val in reward_functions]
        reward_basis_functions = [val[0] for val in reward_functions]

    # Set up blob surveys.
    if bandname2 is None:
        survey_name = "simple pair %i, %s" % (pair_time, bandname)
    else:
        survey_name = "simple pair %i, %s%s" % (pair_time, bandname, bandname2)

    # Set up detailers for each requested observation.
    detailer_list = []
    # Avoid camera rotator limits.
    detailer_list.append(
        detailers.CameraRotDetailer(
            min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits), dither=camera_dither
        )
    )
    # Convert rotTelPos to rotSkyPos_desired
    detailer_list.append(detailers.Rottep2RotspDesiredDetailer(telescope="rubin"))
    # Reorder visits in a blob so that closest to current altitude is first.
    detailer_list.append(detailers.CloseAltDetailer())
    # Sets a flush-by date to avoid running into prescheduled visits.
    detailer_list.append(detailers.FlushForSchedDetailer())
    # Add a detailer to label visits as either first or second of the pair.
    if bandname2 is not None:
        detailer_list.append(detailers.TakeAsPairsDetailer(bandname=bandname2))

    # Set up the survey.
    ignore_obs = ["DD"]

    BlobSurvey_params = {
        "slew_approx": 7.5,
        "band_change_approx": 140.0,
        "read_approx": 2.4,
        "flush_time": pair_time * 3,
        "smoothing_kernel": None,
        "nside": nside,
        "seed": 42,
        "dither": dither,
        "twilight_scale": False,
    }

    pair_survey = BlobSurvey(
        reward_basis_functions + mask_basis_functions,
        reward_basis_functions_weights + mask_basis_functions_weights,
        bandname1=bandname,
        bandname2=bandname2,
        exptime=exptime,
        ideal_pair_time=pair_time,
        survey_name=survey_name,
        observation_reason=observation_reason,
        ignore_obs=ignore_obs,
        nexp=nexp,
        detailers=detailer_list,
        science_program=science_program,
        **BlobSurvey_params,
    )

    # Tucking this here so we can look at how many observations
    # recorded for this survey and what was the last one.
    pair_survey.extra_features["ObsRecorded"] = features.NObsCount()
    pair_survey.extra_features["LastObs"] = features.LastObservation()

    return pair_survey


def simple_greedy_survey(
    nside: int = DEFAULT_NSIDE,
    bandname: str = "r",
    mask_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions_weights: list[float] | None = None,
    survey_start: float = SURVEY_START_MJD,
    footprints_hp: np.ndarray | None = None,
    camera_rot_limits: list[float] = [-80.0, 80.0],
    exptime: float = 30.0,
    nexp: int = 1,
    science_program: str | None = None,
    observation_reason: str | None = None,
    dither: str = "night",
    camera_dither: str = "night",
) -> GreedySurvey:
    """Set up a simple greedy survey to just observe single visits.

    Parameters
    ----------
    nside  : `int`, optional
        Nside for the surveys.
    bandname : `str`, optional
        Bandname for the visits.
    mask_basis_functions : `list` [`BaseBasisFunction`] or None
        List of basis functions to use as masks (with implied weight 0).
        If None, `standard_masks` is used with default parameters.
    reward_basis_functions : `list` [`BaseBasisFunction`] or None
        List of basis functions to use as rewards.
        If None, a basic set of basis functions will be used.
    reward_basis_functions_weights : `list` [`float`] or None
        List of values to use as weights for the reward basis functions.
        If None, default values for the basic set will be used.
    survey_start : `float` or None
        The start of the survey, in MJD.
        If None, `survey_start_mjd()` is used.
        This should be the start of the survey, not the current time.
    footprints_hp : `np.ndarray` (N,) or None
        An array of healpix maps with the target survey area, with dtype
        like [(bandname, '<f8'), (bandname2, '<f8')].
        If None, `get_current_footprint()` will be used, which will cover
        the expected LSST survey footprint.
    camera_rot_limits : `list` [`float`]
        The rotator limits to expect for the camera.
        These should be slightly padded from true limits, to allow for
        slight delays between requesting observations and acquiring them.
    exptime : `float`
        The on-sky exposure time per visit.
    nexp : `int`
        The number of exposures per visit (exptime * nexp = total on-sky time).
    science_program : `str` | None
        The science_program key for the FieldSurvey.
        This maps to the BLOCK and `science_program` in the consDB.
    observation_reason : `str` | None
        The observation_reason for the Survey.
        Indicates survey mode, potentially enhanced with particular reason.

    Returns
    -------
    greedy_survey : `GreedySurvey`
        A greedy survey configured to take the next best (single) visit
        in bandname.
    """

    # Use the Almanac to find the position of the sun at the start of survey.
    almanac = Almanac(mjd_start=survey_start)
    sun_moon_info = almanac.get_sun_moon_positions(survey_start)
    sun_ra_start = sun_moon_info["sun_RA"].copy()

    if footprints_hp is None:
        footprints_hp, labels = get_current_footprint(nside=nside)
    footprints = Footprint(mjd_start=survey_start, sun_ra_start=sun_ra_start, nside=nside)
    for f in footprints_hp.dtype.names:
        footprints.set_footprint(f, footprints_hp[f])

    if mask_basis_functions is None:
        mask_basis_functions = standard_masks(nside=nside)
    # Mask basis functions have zero weights.
    mask_basis_functions_weights = [0 for mask in mask_basis_functions]

    if reward_basis_functions is None:
        # Create list of (tuples of) basic reward basis functions and weights.
        m5_weight = 6.0
        footprint_weight = 1.5
        slewtime_weight = 3.0
        stayband_weight = 3.0
        repeat_weight = -5

        reward_functions = simple_rewards(
            footprints=footprints,
            bandname=bandname,
            nside=nside,
            m5_weight=m5_weight,
            footprint_weight=footprint_weight,
            slewtime_weight=slewtime_weight,
            stayband_weight=stayband_weight,
            repeat_weight=repeat_weight,
        )

        # unpack the basis functions and weights
        reward_basis_functions_weights = [val[1] for val in reward_functions]
        reward_basis_functions = [val[0] for val in reward_functions]

    # Set up survey name, use also for scheduler note.
    survey_name = f"simple greedy {bandname}"

    # Set up detailers for each requested observation.
    detailer_list = []
    # Avoid camera rotator limits.
    detailer_list.append(
        detailers.CameraRotDetailer(
            min_rot=np.min(camera_rot_limits),
            max_rot=np.max(camera_rot_limits),
            dither=camera_dither,
        )
    )
    # Convert rotTelPos to rotSkyPos_desired
    detailer_list.append(detailers.Rottep2RotspDesiredDetailer(telescope="rubin"))
    # Reorder visits in a blob so that closest to current altitude is first.
    detailer_list.append(detailers.CloseAltDetailer())
    # Sets a flush-by date to avoid running into prescheduled visits.
    detailer_list.append(detailers.FlushForSchedDetailer())

    # Set up the survey.
    ignore_obs = ["DD"]

    GreedySurvey_params = {
        "nside": nside,
        "seed": 42,
        "dither": dither,
    }

    greedy_survey = GreedySurvey(
        reward_basis_functions + mask_basis_functions,
        reward_basis_functions_weights + mask_basis_functions_weights,
        bandname=bandname,
        exptime=exptime,
        survey_name=survey_name,
        ignore_obs=ignore_obs,
        nexp=nexp,
        detailers=detailer_list,
        science_program=science_program,
        observation_reason=observation_reason,
        **GreedySurvey_params,
    )

    # Tucking this here so we can look at how many observations
    # recorded for this survey and what was the last one.
    greedy_survey.extra_features["ObsRecorded"] = features.NObsCount()
    greedy_survey.extra_features["LastObs"] = features.LastObservation()

    return greedy_survey


def simple_rewards_field_survey(
    scheduler_note: str,
    nside: int = DEFAULT_NSIDE,
    sun_alt_limit: float = -12.0,
) -> list[basis_functions.BaseBasisFunction]:
    """Get some simple rewards to observe a field survey for a long period.

    Parameters
    ----------
    scheduler_note : `str`
        The scheduler note for the field survey.
        Typically this will be the same as the field name.
    nside : `int`
        The nside value for the healpix grid.
    sun_alt_limit : `float`, optional
        Value for the sun's altitude at which to allow observations to start
        (or finish).

    Returns
    -------
    bfs : `list` of `~.scheduler.basis_functions.BaseBasisFunction`
    """
    bfs = [
        basis_functions.NotTwilightBasisFunction(sun_alt_limit=sun_alt_limit),
        # Avoid revisits within 30 minutes - but we'll have to replace "note"
        basis_functions.VisitGap(band_names=None, note=scheduler_note, gap_min=30.0),
        # reward fields which are rising, but don't mask out after zenith
        basis_functions.RewardRisingBasisFunction(nside=nside, slope=0.1, penalty_val=0),
        # Reward parts of the sky which are darker --
        # note that this is only for r band, so relying on skymap in r band.
        # if there isn't a strong reason to go with the darkest pointing,
        # it might be reasonable to just drop this basis function
        basis_functions.M5DiffBasisFunction(bandname="r", nside=nside),
    ]
    return bfs


def simple_field_survey(
    field_ra_deg: float,
    field_dec_deg: float,
    field_name: str,
    mask_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    detailers: list[detailers.BaseDetailer] | None = None,
    sequence: str | list[str] = "ugrizy",
    nvisits: dict | None = None,
    exptimes: dict | None = None,
    nexps: dict | None = None,
    nside: int = DEFAULT_NSIDE,
    science_program: str | None = None,
) -> FieldSurvey:
    """Set up a simple field survey.

    Parameters
    ----------
    field_ra_deg : `float`
        The RA (in degrees) of the field.
    field_dec_deg : `float`
        The Dec (in degrees) of the field.
    field_name : `str`
        The name of the field. This is used for the survey_name and
        transferred to the 'target' information in the output observation.
        Also used in 'scheduler_note', which is important for the FieldSurvey
        to know whether to count particular observations for the Survey.
    mask_basis_functions : `list` [`BaseBasisFunction`] or None
        List of basis functions to use as masks (with implied weight 0).
        If None, `standard_masks` is used with default parameters.
    reward_basis_functions : `list` [`BaseBasisFunction`] or None
        List of basis functions to use as rewards.
        If None, a basic set of basis functions useful for long observations
        of a field within a night will be used (`get
    detailers : `list` of [`~.scheduler.detailer` objects]
        Detailers for the survey.
        Detailers can add information to output observations, including
        specifying rotator or dither positions.
    sequence : `str` or `list` [`str`]
        The bands (in order?) for the sequence of observations.
    nvisits : `dict` {`str`:`int`} | None
        Number of visits per band to program in the sequence.
        Default of None uses
        nvisits = {"u": 20, "g": 20, "r": 20, "i": 20, "z": 20, "y": 20}
    exptimes : `dict` {`str`: `float`} | None
        Exposure times per band to program in the sequence.
        Default of None uses
        exptimes = {"u": 38, "g": 30, "r": 30, "i": 30, "z": 30, "y": 30}
    nexps : `dict` {`str`: `int`} | None
        Number of exposures per band to program in the sequence.
        Default of None uses
        nexps = {"u": 1, "g": 2, "r": 2, "i": 2, "z": 2, "y": 2}
    nside : `int`, optional
        Nside for the survey. Default DEFAULT_NSIDE.
    science_program : `str` | None
        The science_program key for the FieldSurvey.
        This maps to the BLOCK and `science_program` in the consDB.

    Returns
    -------
    field_survey : `~.scheduler.surveys.FieldSurvey`
        The configured FieldSurvey.

    Notes
    -----
    The sequences for a given field survey can be set via kwargs,
    not necessarily easily accessible here. Only portions of the sequence
    which correspond to mounted bands will be requested by the FieldSurvey.

    field_survey.extra_features['ObsRecord'] tracks how many observations
    have been accepted by the Survey (and can be useful for diagnostics).
    """
    if mask_basis_functions is None:
        mask_basis_functions = standard_masks(nside=nside)
    if reward_basis_functions is None:
        reward_basis_functions = simple_rewards_field_survey(field_name, nside=nside)
    basis_functions = mask_basis_functions + reward_basis_functions

    if nvisits is None:
        nvisits = {"u": 20, "g": 20, "r": 20, "i": 20, "z": 20, "y": 20}
    if exptimes is None:
        exptimes = {"u": 38, "g": 30, "r": 30, "i": 30, "z": 30, "y": 30}
    if nexps is None:
        nexps = {"u": 1, "g": 2, "r": 2, "i": 2, "z": 2, "y": 2}

    field_survey = FieldSurvey(
        basis_functions,
        field_ra_deg,
        field_dec_deg,
        sequence=sequence,
        nvisits=nvisits,
        exptimes=exptimes,
        nexps=nexps,
        ignore_obs=None,
        survey_name=field_name,
        scheduler_note=field_name,
        target_name=field_name,
        readtime=2.4,
        band_change_time=120.0,
        nside=nside,
        flush_pad=30.0,
        detailers=detailers,
        science_program=science_program,
    )
    return field_survey
