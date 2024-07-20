__all__ = (
    "get_model_observatory",
    "update_model_observatory_sunset",
    "get_basis_functions_field_survey",
    "get_field_survey",
    "get_sv_fields",
    "prioritize_fields",
    "get_comcam_sv_schedulers",
)

import copy

import numpy as np
from astropy.time import Time

from rubin_scheduler.scheduler import basis_functions
from rubin_scheduler.scheduler.detailers import CameraSmallRotPerObservationListDetailer
from rubin_scheduler.scheduler.model_observatory import (
    KinemModel,
    ModelObservatory,
    rotator_movement,
    tma_movement,
)
from rubin_scheduler.scheduler.schedulers import ComCamFilterSched, CoreScheduler
from rubin_scheduler.scheduler.surveys import FieldSurvey
from rubin_scheduler.site_models import ConstantSeeingData, ConstantWindData


def get_model_observatory(
    dayobs="2024-09-09",
    fwhm_500=2.0,
    wind_speed=5.0,
    wind_direction=340,
    tma_percent=10,
    rotator_percent=100,
    survey_start=Time("2024-09-09T12:00:00", format="isot", scale="utc").mjd,
):
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


def update_model_observatory_sunset(observatory, filter_scheduler, twilight=-12):
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


def get_field_survey(field_ra_deg, field_dec_deg, field_name, basis_functions, detailers, nside=32):
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


def get_sv_fields():
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


def prioritize_fields(priority_fields=None, field_dict=None):
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


def get_comcam_sv_schedulers(starting_tier=0, tiers=None, field_dict=None, nside=32):
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
