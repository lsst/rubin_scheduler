__all__ = ["gen_roman_on_season", "gen_roman_off_season"]

import copy

import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.surveys import DeepDrillingSurvey
from rubin_scheduler.utils import DEFAULT_NSIDE, special_locations

from .lsst_surveys import EXPTIME, NEXP, SCIENCE_PROGRAM, safety_masks


def def_roman_info() -> dict:
    """Define some rough estimates for Roman RGES on-sky info.

    Returns
    -------
    roman_info : `dict`
        Dictionary containing field location and expected observing seasons.
    """
    # From the TVS Slack channel:
    # spring 2027, fall 2027, spring 2028, then fall 2030,
    # spring 2031, fall 2031

    roman_info = {}
    locations = special_locations()

    roman_info["RA"] = locations["Roman_bulge_location"][0]
    roman_info["dec"] = locations["Roman_bulge_location"][1]

    # Guessing these from the notebook in same dir.
    observing_season_mid_mjds = [61947.3, 62318.8, 62670.3, 63067.2, 63381.4, 63773.2]

    roman_info["seasons_on"] = [[val - 32, val + 32] for val in observing_season_mid_mjds]

    roman_info["seasons_off"] = []
    for i in range(len(roman_info["seasons_on"]) - 1):
        roman_info["seasons_off"].append(
            [roman_info["seasons_on"][i][1] + 1, roman_info["seasons_on"][i + 1][0] - 1]
        )

    return roman_info


def gen_roman_on_season(
    nside: int = DEFAULT_NSIDE,
    max_dither: float = 0.2,
    per_night: bool = False,
    camera_ddf_rot_limit: float = 75.0,
    camera_ddf_rot_per_visit: float = 2.0,
    exptime: float = EXPTIME,
    nexp: int = NEXP,
    science_program: str = SCIENCE_PROGRAM,
    safety_mask_params: dict | None = None,
) -> DeepDrillingSurvey:
    """Generate a survey configured to observe the Roman field(s) during an
    'on' season.

    Parameters
    ----------
    nside : `int`
        The HEALpix nside for the survey, used for basis functions.
    max_dither : `float`
        The maximum radial offset for translation (RA/Dec) dithers.
        In degrees.
    per_night : `bool`
        Whether to dither per_night (True) or for each visit (False).
    camera_ddf_rot_limit : `float`
        The rotator limits for the DDF field rotational dithers.
        In degrees.
    camera_ddf_rot_per_visit : `float`
        The rotational dither offset to apply per visit.
        In degrees.
    exptime : `float`
        The exposure time per visit.
    nexp : `int`
        The number of snaps (exposures) per visit.
    science_program : `str`
        Name of the science program for the survey.
    safety_mask_params : `dict`
        A dictionary of additional kwargs to pass to the standard safety masks.

    Returns
    -------
    survey : `DeepDrillingSurvey`
        A survey configured to observe in a sequence of 'giriz'
        (single exposure each band), every day while the RGES field
        is being observed by Roman.
    """
    if safety_mask_params is None:
        safety_mask_params = {}
        safety_mask_params["nside"] = nside
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)
    # Estimate of sequence time, given sequence defined below.
    shadow_minutes = 12
    if "shadow_minutes" not in safety_mask_params or safety_mask_params["shadow_minutes"] < shadow_minutes:
        safety_mask_params["shadow_minutes"] = shadow_minutes

    field_info = def_roman_info()

    RA = field_info["RA"]
    dec = field_info["dec"]

    survey_name = "DD: RGES_onseason"

    # Add some feasability basis functions.
    basis_functions = safety_masks(**safety_mask_params)
    # These are crude hard limits. Nominally we would try to
    # pre-schedule these when they would be at the best airamss
    # in the night.
    basis_functions.append(bf.HourAngleLimitBasisFunction(RA=RA, ha_limits=[[20, 24], [0, 4]]))
    basis_functions.append(bf.NotTwilightBasisFunction())
    # Force it to delay about a day
    basis_functions.append(bf.ForceDelayBasisFunction(days_delay=0.8, scheduler_note=survey_name))
    # Force it to be in a given observing season
    basis_functions.append(bf.InTimeWindowBasisFunction(mjd_windows=field_info["seasons_on"]))

    # Add a dither detailer, so it dithers between each set of exposures
    details = []
    details.append(detailers.DitherDetailer(max_dither=max_dither, per_night=per_night))
    details.append(
        detailers.CameraSmallRotPerObservationListDetailer(
            min_rot=-camera_ddf_rot_limit,
            max_rot=camera_ddf_rot_limit,
            per_visit_rot=camera_ddf_rot_per_visit,
        ),
    )
    # Don't sort the bands, but at least start with the one in use.
    details.append(detailers.RollBandMatchDetailer())
    details.append(
        detailers.TrackingInfoDetailer(
            target_name="roman_field",
            observation_reason="rges_onseason",
            science_program=science_program,
        )
    )
    details.append(detailers.LabelRegionsAndDDFs())

    survey = DeepDrillingSurvey(
        basis_functions,
        RA=RA,
        dec=dec,
        sequence="giriz",
        # This may need some work if rapid filter changes cause problems.
        # However, this survey doesn't execute until RGES is in season (2028?)
        nvis=[1, 1, 1, 1, 1],
        exptime=exptime,
        nexp=nexp,
        survey_name=survey_name,
        detailers=details,
    )
    return survey


def gen_roman_off_season(
    nside: int = DEFAULT_NSIDE,
    max_dither: float = 0.2,
    per_night: bool = False,
    camera_ddf_rot_limit: float = 75.0,
    camera_ddf_rot_per_visit: float = 2.0,
    exptime: float = EXPTIME,
    nexp: int = NEXP,
    science_program: str = SCIENCE_PROGRAM,
    safety_mask_params: dict | None = None,
) -> DeepDrillingSurvey:
    """Generate a survey configured to observe the Roman field(s) outside
    of the 'on' seasons (during 'off' seasons).

    Parameters
    ----------
    nside : `int`
        The HEALpix nside for the survey, used for basis functions.
    max_dither : `float`
        The maximum radial offset for translation (RA/Dec) dithers.
        In degrees.
    per_night : `bool`
        Whether to dither per_night (True) or for each visit (False).
    camera_ddf_rot_limit : `float`
        The rotator limits for the DDF field rotational dithers.
        In degrees.
    camera_ddf_rot_per_visit : `float`
        The rotational dither offset to apply per visit.
        In degrees.
    exptime : `float`
        The exposure time per visit.
    nexp : `int`
        The number of snaps (exposures) per visit.
    science_program : `str`
        Name of the science program for the survey.
    safety_mask_params : `dict`
        A dictionary of additional kwargs to pass to the standard safety masks.

    Returns
    -------
    survey : `DeepDrillingSurvey`
        A survey configured to observe in a sequence of 'griz'
        every third day while the RGES field is visible but not being
        observed by Roman.
    """
    if safety_mask_params is None:
        safety_mask_params = {}
        safety_mask_params["nside"] = nside
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)
    # Estimate of sequence time, given sequence defined below.
    shadow_minutes = 11
    if "shadow_minutes" not in safety_mask_params or safety_mask_params["shadow_minutes"] < shadow_minutes:
        safety_mask_params["shadow_minutes"] = shadow_minutes

    field_info = def_roman_info()
    RA = field_info["RA"]
    dec = field_info["dec"]

    survey_name = "DD: RGES_offseason"

    # Add some feasability basis functions. Maybe just give it a
    # set of nights where it can execute for now.
    basis_functions = safety_masks(**safety_mask_params)
    # These are crude hard limits. Nominally we would try
    # to pre-schedule these when they would be at the best
    # airamss in the night.
    basis_functions.append(bf.HourAngleLimitBasisFunction(RA=RA, ha_limits=[[20, 24], [0, 4]]))
    basis_functions.append(bf.NotTwilightBasisFunction())
    # Force it to not go every day
    basis_functions.append(bf.ForceDelayBasisFunction(days_delay=3.0, scheduler_note=survey_name))
    # Force it to be in a given observing season
    basis_functions.append(bf.InTimeWindowBasisFunction(mjd_windows=field_info["seasons_off"]))
    basis_functions.append(bf.AirmassPointRangeBasisFunction(RA, dec, nside=nside))

    # Add a dither detailer, so it dithers between each set of exposures
    details = []
    details.append(detailers.DitherDetailer(max_dither=max_dither, per_night=per_night))
    details.append(
        detailers.CameraSmallRotPerObservationListDetailer(
            min_rot=-camera_ddf_rot_limit,
            max_rot=camera_ddf_rot_limit,
            per_visit_rot=camera_ddf_rot_per_visit,
        ),
    )
    # Don't sort the bands, but at least start with the one in use.
    details.append(detailers.RollBandMatchDetailer())
    details.append(
        detailers.TrackingInfoDetailer(
            target_name="roman_field",
            observation_reason="rges_offseason",
            science_program=science_program,
        )
    )
    details.append(detailers.LabelRegionsAndDDFs())

    survey = DeepDrillingSurvey(
        basis_functions,
        RA=RA,
        dec=dec,
        sequence="griz",
        # updated these to 2 visits before filter change
        # original plan is 1 per band -- can AOS handle this?
        nvis=[2, 2, 2, 2],
        exptime=exptime,
        nexp=nexp,
        survey_name=survey_name,
        detailers=details,
    )
    return survey
