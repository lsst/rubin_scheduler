__all__ = (
    "get_model_observatory",
    "update_model_observatory_sunset",
    "standard_masks",
    "simple_rewards",
    "simple_pairs_survey",
    "simple_greedy_survey",
    "get_basis_functions_field_survey",
    "get_field_survey",
    "get_sv_fields",
    "prioritize_fields",
    "get_comcam_sv_schedulers",
)

import copy

import numpy as np
from astropy.time import Time

import rubin_scheduler.scheduler.basis_functions as basis_functions
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.detailers import CameraSmallRotPerObservationListDetailer
from rubin_scheduler.scheduler.model_observatory import (
    KinemModel,
    ModelObservatory,
    rotator_movement,
    tma_movement,
)
from rubin_scheduler.scheduler.schedulers import ComCamFilterSched, CoreScheduler, FilterSwapScheduler
from rubin_scheduler.scheduler.surveys import BlobSurvey, FieldSurvey, GreedySurvey
from rubin_scheduler.scheduler.utils import Footprint, get_current_footprint
from rubin_scheduler.site_models import Almanac, ConstantSeeingData, ConstantWindData
from rubin_scheduler.utils import survey_start_mjd


def get_model_observatory(
    dayobs: str = "2024-09-09",
    fwhm_500: float = 2.0,
    wind_speed: float = 5.0,
    wind_direction: float = 340,
    tma_percent: float = 10,
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
    dayobs : `str`
        DAYOBS formatted str (YYYY-MM-DD) for the evening to start
        up the observatory.
    fwhm_500 : `float`, optional
        The value to set for atmospheric component of seeing,
         constant seeing throughout the night (arcseconds).
        Ad-hoc default for start of comcam on-sky operations about 2.0".
    wind_speed : `float`, optional
        Set a (constant) wind speed for the night, (m/s).
        Default of 5.0 is minimal but noticeable.
    wind_direction : `float`, optional
        Set a (constant) wind direction for the night (deg).
        Default of 340 is midrange between typical wind directions for
        the site (see SITCOMTN-126).
    tma_percent : `float`, optional
        Set a percent of full-performance for the telescope TMA (0-100).
        Default of 10(%) is likely for start of comcam on-sky SV surveys.
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
    observatory. The filters may not be correct however; use
    `update_model_observatory_sunset` to get filters in place.
    """
    # Set up a fresh model observatory
    mjd_now = Time(f"{dayobs}T12:00:00", format="isot", scale="utc").mjd

    kinematic_model = KinemModel(mjd0=mjd_now)
    rot = rotator_movement(rotator_percent)
    kinematic_model.setup_camera(readtime=2.4, **rot)
    tma = tma_movement(tma_percent)
    kinematic_model.setup_telescope(**tma)

    # Some weather telemetry that might be useful
    seeing_data = ConstantSeeingData(fwhm_500=fwhm_500)
    wind_data = ConstantWindData(wind_direction=wind_direction, wind_speed=wind_speed)

    # Set up the model observatory
    observatory = ModelObservatory(
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
    observatory: ModelObservatory, filter_scheduler: FilterSwapScheduler, twilight: int | float = -12
) -> ModelObservatory:
    """Move model observatory to twilight and ensure correct filters are in
    place according to the filter_scheduler.

    Parameters
    ----------
    observatory : `~.scheduler.model_observatory.ModelObservatory`
        The ModelObservatory simulating the observatory.
    filter_scheduler : `~.scheduler.schedulers.FilterScheduler`
        The filter scheduler providing appropriate information on
        the filters that should be in place on the current observatory day.
    twilight : `int` or `float`
        If twilight is -12 or -18, the Almanac -12 or -18 degree twilight
        times are used to set the current observatory time.
        If  any other value is provided, it is assumed to be a specific
        MJD to start operating the observatory.
        Filter choices are based on the time after advancing to twilight.

    Returns
    -------
    observatory :  `~.scheduler.model_observatory.ModelObservatory`
        The ModelObservatory simulating the observatory, updated to the
        time of 'twilight' and with mounted_filters matching
        the filters chosen by the filter_scheduler for the current time
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

    # Make sure correct filters are mounted
    conditions = observatory.return_conditions()
    filters_needed = filter_scheduler(conditions)
    observatory.observatory.mount_filters(filters_needed)
    return observatory


def standard_masks(
    nside: int,
    moon_distance: float = 30.0,
    wind_speed_maximum: float = 20.0,
    min_alt: float = 20,
    max_alt: float = 86.5,
    shadow_minutes: float = 30,
) -> list[basis_functions.BaseBasisFunction]:
    """A set of standard mask functions.

    Avoids the moon, high wind, and areas on the sky out of bounds via
    the slew calculation from the model observatory.

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
            min_az=0.0,
            max_az=360.0,
            shadow_minutes=shadow_minutes,
        )
    )
    return masks


def simple_rewards(
    footprints: Footprint,
    filtername: str,
    nside: int = 32,
    m5_weight: float = 6.0,
    footprint_weight: float = 1.5,
    slewtime_weight: float = 3.0,
    stayfilter_weight: float = 3.0,
    repeat_weight: float = -20,
) -> list[basis_functions.BaseBasisFunction]:
    """A simple set of rewards for area-based surveys.

    Parameters
    ----------
    footprints : `Footprint`
        A Footprint class, which takes a target map and adds a
        time-dependent weighting on top (such as seasonal weighting).
    filtername : `str`
        The filtername active for these rewards.
    nside : `int`, optional
        The nside for the rewards.
    m5_weight : `float`, optional
        The weight to give the M5Diff basis function.
    footprint_weight : `float`, optional
        The weight to give the footprint basis function.
    slewtime_weight : `float`, optional
        The weight to give the slewtime basis function.
    stayfilter_weight : `float`, optional
        The weight to give the FilterChange basis function.
    repeat_weight : `float`, optional
        The weight to give the VisitRepeat basis function.
        This is negative by default, to avoid revisiting the same part
        of the sky within `gap_max` (3 hours) in any filter (except in the
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
    reward_functions.append(
        (basis_functions.M5DiffBasisFunction(filtername=filtername, nside=nside), m5_weight)
    )
    # Add a footprint basis function
    # (rewards sky with fewer visits)
    reward_functions.append(
        (
            basis_functions.FootprintBasisFunction(
                filtername=filtername,
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
            basis_functions.SlewtimeBasisFunction(filtername=filtername, nside=nside),
            slewtime_weight,
        )
    )
    # Add a reward to stay in the same filter as much as possible.
    reward_functions.append(
        (basis_functions.FilterChangeBasisFunction(filtername=filtername), stayfilter_weight)
    )
    # And this is technically a mask, to avoid asking for observations
    # which are not possible. However, it depends on the filters
    # requested, compared to the filters available in the camera.
    reward_functions.append((basis_functions.FilterLoadedBasisFunction(filternames=filtername), 0))
    # And add a basis function to avoid repeating the same pointing
    # (VisitRepeat can be used to either encourage or discourage repeats,
    # depending on the repeat_weight value).
    reward_functions.append(
        (
            basis_functions.VisitRepeatBasisFunction(
                gap_min=0, gap_max=3 * 60.0, filtername=None, nside=nside, npairs=20
            ),
            repeat_weight,
        )
    )
    return reward_functions


def simple_pairs_survey(
    nside: int = 32,
    filtername: str = "g",
    filtername2: str | None = None,
    mask_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions_weights: list[float] | None = None,
    survey_start: float | None = None,
    footprints_hp: np.ndarray | None = None,
    camera_rot_limits: list[float] = [-80.0, 80.0],
    pair_time: float = 30.0,
    exptime: float = 30.0,
    nexp: int = 1,
) -> BlobSurvey:
    """Set up a simple blob survey to acquire pairs of visits.

    Parameters
    ----------
    nside  : `int`, optional
        Nside for the surveys.
    filtername : `str`, optional
        Filtername for the first visit of the pair.
    filtername2 : `str` or None, optional
        Filtername for the second visit of the pair. If None, the
        first filter will be used for both visits.
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
        like [(filtername, '<f8'), (filtername2, '<f8')].
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

    Returns
    -------
    pair_survey : `BlobSurvey`
        A blob survey configured to take pairs at spacing of pair_time,
        in filtername + filtername2.
    """

    # Note that survey_start should be the start of the FIRST night of survey.
    # Not the current night.
    if survey_start is None:
        survey_start = survey_start_mjd()

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
        stayfilter_weight = 3.0
        repeat_weight = -20

        if filtername2 is None:
            reward_functions = simple_rewards(
                footprints=footprints,
                filtername=filtername,
                nside=nside,
                m5_weight=m5_weight,
                footprint_weight=footprint_weight,
                slewtime_weight=slewtime_weight,
                stayfilter_weight=stayfilter_weight,
                repeat_weight=repeat_weight,
            )

        else:
            # Add the same basis functions, but M5 and footprint
            # basis functions need to be added twice, with half the weight.
            rf1 = simple_rewards(
                footprints=footprints,
                filtername=filtername,
                nside=nside,
                m5_weight=m5_weight / 2.0,
                footprint_weight=footprint_weight / 2.0,
                slewtime_weight=slewtime_weight,
                stayfilter_weight=stayfilter_weight,
                repeat_weight=repeat_weight,
            )
            rf2 = simple_rewards(
                footprints=footprints,
                filtername=filtername2,
                nside=nside,
                m5_weight=m5_weight / 2.0,
                footprint_weight=footprint_weight / 2.0,
                slewtime_weight=0,
                stayfilter_weight=0,
                repeat_weight=0,
            )
            # Now clean up and combine these - and remove the separate
            # BasisFunction for FilterLoadedBasisFunction.
            reward_functions = [(i[0], i[1]) for i in rf1 if i[1] > 0] + [
                (i[0], i[1]) for i in rf2 if i[1] > 0
            ]
            # Then put back in the FilterLoadedBasisFunction with both filters.
            filternames = [fn for fn in [filtername, filtername2] if fn is not None]
            reward_functions.append((basis_functions.FilterLoadedBasisFunction(filternames=filternames), 0))

        # unpack the basis functions and weights
        reward_basis_functions_weights = [val[1] for val in reward_functions]
        reward_basis_functions = [val[0] for val in reward_functions]

    # Set up blob surveys.
    if filtername2 is None:
        scheduler_note = "pair_%i, %s" % (pair_time, filtername)
    else:
        scheduler_note = "pair_%i, %s%s" % (pair_time, filtername, filtername2)

    # Set up detailers for each requested observation.
    detailer_list = []
    # Avoid camera rotator limits.
    detailer_list.append(
        detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
    )
    # Convert rotTelPos to rotSkyPos_desired
    detailer_list.append(detailers.Rottep2RotspDesiredDetailer(telescope="rubin"))
    # Reorder visits in a blob so that closest to current altitude is first.
    detailer_list.append(detailers.CloseAltDetailer())
    # Sets a flush-by date to avoid running into prescheduled visits.
    detailer_list.append(detailers.FlushForSchedDetailer())
    # Add a detailer to label visits as either first or second of the pair.
    if filtername2 is not None:
        detailer_list.append(detailers.TakeAsPairsDetailer(filtername=filtername2))

    # Set up the survey.
    ignore_obs = ["DD"]

    BlobSurvey_params = {
        "slew_approx": 7.5,
        "filter_change_approx": 140.0,
        "read_approx": 2.4,
        "search_radius": 30.0,
        "flush_time": pair_time * 3,
        "smoothing_kernel": None,
        "nside": nside,
        "seed": 42,
        "dither": True,
        "twilight_scale": False,
    }

    pair_survey = BlobSurvey(
        reward_basis_functions + mask_basis_functions,
        reward_basis_functions_weights + mask_basis_functions_weights,
        filtername1=filtername,
        filtername2=filtername2,
        exptime=exptime,
        ideal_pair_time=pair_time,
        scheduler_note=scheduler_note,
        ignore_obs=ignore_obs,
        nexp=nexp,
        detailers=detailer_list,
        **BlobSurvey_params,
    )

    return pair_survey


def simple_greedy_survey(
    nside: int = 32,
    filtername: str = "r",
    mask_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions: list[basis_functions.BaseBasisFunction] | None = None,
    reward_basis_functions_weights: list[float] | None = None,
    survey_start: float | None = None,
    footprints_hp: np.ndarray | None = None,
    camera_rot_limits: list[float] = [-80.0, 80.0],
    exptime: float = 30.0,
    nexp: int = 1,
) -> GreedySurvey:
    """Set up a simple greedy survey to just observe single visits.

    Parameters
    ----------
    nside  : `int`, optional
        Nside for the surveys.
    filtername : `str`, optional
        Filtername for the visits.
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
        like [(filtername, '<f8'), (filtername2, '<f8')].
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

    Returns
    -------
    greedy_survey : `GreedySurvey`
        A greedy survey configured to take the next best visit
        in filtername.
    """

    # Note that survey_start should be the start of the FIRST night of survey.
    # Not the current night.
    if survey_start is None:
        survey_start = survey_start_mjd()

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
        mask_basis_functions = standard_masks(nside=nside, shadow_minutes=30)
    # Mask basis functions have zero weights.
    mask_basis_functions_weights = [0 for mask in mask_basis_functions]

    if reward_basis_functions is None:
        # Create list of (tuples of) basic reward basis functions and weights.
        m5_weight = 6.0
        footprint_weight = 1.5
        slewtime_weight = 3.0
        stayfilter_weight = 3.0
        repeat_weight = -5

        reward_functions = simple_rewards(
            footprints=footprints,
            filtername=filtername,
            nside=nside,
            m5_weight=m5_weight,
            footprint_weight=footprint_weight,
            slewtime_weight=slewtime_weight,
            stayfilter_weight=stayfilter_weight,
            repeat_weight=repeat_weight,
        )

        # unpack the basis functions and weights
        reward_basis_functions_weights = [val[1] for val in reward_functions]
        reward_basis_functions = [val[0] for val in reward_functions]

    # Set up scheduler note.
    scheduler_note = f"greedy {filtername}"

    # Set up detailers for each requested observation.
    detailer_list = []
    # Avoid camera rotator limits.
    detailer_list.append(
        detailers.CameraRotDetailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))
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
        "dither": True,
        "block_size": 2,
    }

    greedy_survey = GreedySurvey(
        reward_basis_functions + mask_basis_functions,
        reward_basis_functions_weights + mask_basis_functions_weights,
        filtername=filtername,
        exptime=exptime,
        scheduler_note=scheduler_note,
        ignore_obs=ignore_obs,
        nexp=nexp,
        detailers=detailer_list,
        **GreedySurvey_params,
    )

    return greedy_survey


def get_basis_functions_field_survey(
    nside: int = 32,
    wind_speed_maximum: float = 10,
) -> list[basis_functions.BaseBasisFunction]:
    """Get the basis functions for a comcam SV field survey.

    Parameters
    ----------
    nside : `int`
        The nside value for the healpix grid.
    wind_speed_maximum : `float`
        Maximum wind speed tolerated for the observations of the survey,
        in m/s.

    Returns
    -------
    bfs : `list` of `~.scheduler.basis_functions.BaseBasisFunction`
    """
    sun_alt_limit = -12.0
    moon_distance = 30

    bfs = [
        basis_functions.NotTwilightBasisFunction(sun_alt_limit=sun_alt_limit),
        basis_functions.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance),
        basis_functions.AvoidDirectWind(wind_speed_maximum=wind_speed_maximum, nside=nside),
        # Mask parts of the sky in alt/az, including parts of the sky that
        # will move into this area
        # (replaces azimuth mask and zenith shadow mask,
        # should also be able to replace airmass basis function)
        basis_functions.AltAzShadowMaskBasisFunction(
            nside=nside, min_alt=22, max_alt=83, min_az=0.0, max_az=360.0, shadow_minutes=30
        ),
        # Avoid revisits within 30 minutes
        basis_functions.AvoidFastRevisitsBasisFunction(nside=nside, filtername=None, gap_min=30.0),
        # reward fields which are rising, but don't mask out after zenith
        basis_functions.RewardRisingBasisFunction(nside=nside, slope=0.1, penalty_val=0),
        # Reward parts of the sky which are darker --
        # note that this is only for r band, so relying on skymap in r band
        # .. if there isn't a strong reason to go with the darkest pointing,
        # it might be reasonable to just drop this basis function
        basis_functions.M5DiffBasisFunction(filtername="r", nside=nside),
    ]
    return bfs


def get_field_survey(
    field_ra_deg: float,
    field_dec_deg: float,
    field_name: str,
    basis_functions: list[basis_functions.BaseBasisFunction],
    detailers: list[detailers.BaseDetailer],
    nside: int = 32,
) -> FieldSurvey:
    """Set up a comcam SV field survey.

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
    basis_functions : `list` of [`~.scheduler.basis_function` objects]
        Basis functions for the field survey.
        A default set can be obtained from `get_basis_functions_field_survey`.
    detailers : `list` of [`~.scheduler.detailer` objects]
        Detailers for the survey.
        Detailers can add information to output observations, including
        specifying rotator or dither positions.
    nside : `int`, optional
        Nside for the survey. Default 32.

    Returns
    -------
    field_survey : `~.scheduler.surveys.FieldSurvey`
        The configured FieldSurvey.

    Notes
    -----
    The sequences for a given field survey can be set via kwargs,
    not necessarily easily accessible here. Only portions of the sequence
    which correspond to mounted filters will be requested by the FieldSurvey.

    field_survey.extra_features['ObsRecord'] tracks how many observations
    have been accepted by the Survey (and can be useful for diagnostics).
    """
    field_survey = FieldSurvey(
        basis_functions,
        field_ra_deg,
        field_dec_deg,
        sequence="ugrizy",
        nvisits={"u": 20, "g": 20, "r": 20, "i": 20, "z": 20, "y": 20},
        exptimes={"u": 38, "g": 30, "r": 30, "i": 30, "z": 30, "y": 30},
        nexps={"u": 1, "g": 2, "r": 2, "i": 2, "z": 2, "y": 2},
        ignore_obs=None,
        accept_obs=[field_name],
        survey_name=field_name,
        scheduler_note=None,
        readtime=2.4,
        filter_change_time=120.0,
        nside=nside,
        flush_pad=30.0,
        detailers=detailers,
    )
    return field_survey


def get_sv_fields() -> dict[str, dict[str, float]]:
    """Default potential fields for the SV surveys.

    Returns
    -------
    fields_dict : `dict`  {`str` : {'RA' : `float`, 'Dec' : `float`}}
        A dictionary keyed by field_name, containing RA and Dec (in degrees)
        for each field.
    """
    fields = (
        ("Rubin_SV_095_-25", 95.0, -25.0),  # High stellar densty, low extinction
        ("Rubin_SV_125_-15", 125.0, -15.0),  # High stellar densty, low extinction
        ("DESI_SV3_R1", 179.60, 0.000),  # DESI, GAMA, HSC DR2, KiDS-N
        ("Rubin_SV_225_-40", 225.0, -40.0),  # 225 High stellar densty, low extinction
        ("DEEP_A0", 216, -12.5),  # DEEP Solar Systen
        ("Rubin_SV_250_2", 250.0, 2.0),  # 250 High stellar densty, low extinction
        ("Rubin_SV_300_-41", 300.0, -41.0),  # High stellar densty, low extinction
        ("Rubin_SV_280_-48", 280.0, -48.0),  # High stellar densty, low extinction
        ("DEEP_B0", 310, -19),  # DEEP Solar System
        ("ELAIS_S1", 9.45, -44.0),  # ELAIS-S1 LSST DDF
        ("XMM_LSS", 35.708333, -4.75),  # LSST DDF
        ("ECDFS", 53.125, -28.1),  # ECDFS
        ("COSMOS", 150.1, 2.1819444444444445),  # COSMOS
        ("EDFS_A", 58.9, -49.315),  # EDFS_a
        ("EDFS_B", 63.6, -47.6),  # EDFS_b
    )

    fields_dict = dict(zip([f[0] for f in fields], [{"RA": f[1], "Dec": f[2]} for f in fields]))
    return fields_dict


def prioritize_fields(
    priority_fields: list[str] | None = None, field_dict: dict[str, dict[str, float]] | None = None
) -> list[list[FieldSurvey]]:
    """Add the remaining field names in field_dict into the last
    tier of 'priority_fields' field names, creating a complete
    survey tier list of lists.

    Parameters
    ----------
    priority_fields : `list` [`list`]
        A list of lists, where each final list corresponds to a 'tier'
        of FieldSurveys, and contains those field survey names.
        These names must be present in field_dict.
    field_dict :  `dict`  {`str` : {'RA' : `float`, 'Dec' : `float`}} or None
        Dictionary containing field information for the FieldSurveys.
        Default None will fetch the SV fields from 'get_sv_fields'.

    Returns
    -------
    tiers : `list` [`list`]
        The tiers to pass to the core scheduler, after including the
        non-prioritized fields from field_dict.
    """
    if field_dict is None:
        field_dict = get_sv_fields()
    else:
        field_dict = copy.deepcopy(field_dict)
    tiers = []
    if priority_fields is not None:
        for tier in priority_fields:
            tiers.append(tier)
            for field in tier:
                del field_dict[field]
    remaining_fields = list(field_dict.keys())
    tiers.append(remaining_fields)
    return tiers


def get_comcam_sv_schedulers(
    starting_tier: int = 0,
    tiers: list[list[str]] | None = None,
    field_dict: dict[str, dict[str, float]] | None = None,
    nside: int = 32,
) -> (CoreScheduler, ComCamFilterSched):
    """Set up a CoreScheduler and FilterScheduler generally
    appropriate for ComCam SV observing.

    Parameters
    ----------
    starting_tier : `int`, optional
        Starting to tier to place the surveys coming from the 'tiers'
        specified here.
        Default 0, to start at first tier. If an additional
        survey will be added at highest tier after (such as cwfs), then
        set starting tier to 1+ and add these surveys as a list to
        scheduler.survey_lists[tier] etc.
    tiers : `list` [`str`] or None
        Field names for each of the field surveys in tiers.
        Should be a list of lists - [tier][surveys_in_tier]
        [[field1, field2],[field3, field4, field5].
        Fields should be present in the 'field_dict'.
        Default None will use all fields in field_dict.
    field_dict :  `dict`  {`str` : {'RA' : `float`, 'Dec' : `float`}} or None
        Dictionary containing field information for the FieldSurveys.
        Default None will fetch the SV fields from 'get_sv_fields'.
    nside : `int`
        Nside for the scheduler. Default 32.
        Generally, don't change without serious consideration.

    Returns
    -------
    scheduler, filter_scheduler : `~.scheduler.schedulers.CoreScheduler`,
                                  `~.scheduler.schedulers.ComCamFilterSched`
          A CoreScheduler and FilterScheduler that are generally
          appropriate for ComCam.
    """
    if field_dict is None:
        field_dict = get_sv_fields()

    if tiers is None:
        tiers = [list(field_dict.keys())]

    surveys = []

    i = 0
    for t in tiers:
        if len(t) == 0:
            continue
        j = i + starting_tier
        i += 1
        surveys.append([])
        for kk, fieldname in enumerate(t):
            bfs = get_basis_functions_field_survey()
            detailer = CameraSmallRotPerObservationListDetailer(per_visit_rot=0.5)
            surveys[j].append(
                get_field_survey(
                    field_dict[fieldname]["RA"], field_dict[fieldname]["Dec"], fieldname, bfs, [detailer]
                )
            )

    scheduler = CoreScheduler(surveys, nside=nside)
    filter_scheduler = ComCamFilterSched()
    return scheduler, filter_scheduler
