__all__ = ["gen_roman_on_season", "gen_roman_off_season"]

import warnings

import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.surveys import DeepDrillingSurvey
from rubin_scheduler.utils import DEFAULT_NSIDE, special_locations


def roman_info():
    """Manually enter some Roman RGES info."""
    # From the TVS Slack channel:
    # spring 2027, fall 2027, spring 2028, then fall 2030,
    # spring 2031, fall 2031

    result = {}
    locations = special_locations()

    result["RA"] = locations["Roman_bulge_location"][0]
    result["dec"] = locations["Roman_bulge_location"][1]

    # Guessing these from the notebook in same dir.
    observing_season_mid_mjds = [61947.3, 62318.8, 62670.3, 63067.2, 63381.4, 63773.2]

    result["seasons_on"] = [[val - 32, val + 32] for val in observing_season_mid_mjds]

    result["seasons_off"] = []
    for i in range(len(result["seasons_on"]) - 1):
        result["seasons_off"].append([result["seasons_on"][i][1] + 1, result["seasons_on"][i + 1][0] - 1])

    return result


def gen_roman_on_season(
    nside=DEFAULT_NSIDE,
    camera_ddf_rot_limit=75.0,
    exptime=30.0,
    nexp=2,
    wind_speed_maximum=20.0,
    moon_limit=30.0,
    science_program=None,
):
    """Generate a survey object for observing the Roman field(s)
    in an on season"""

    warnings.warn("Fucntion gen_roman_on_season moving out of rubin_scheduler.", DeprecationWarning)
    warnings.warn("Generating Roman survey place holder. Should probably not be in production.")

    field_info = roman_info()

    RA = field_info["RA"]
    dec = field_info["dec"]

    scheduler_note = "DD: RGES_onseason"

    # Add some feasability basis functions. Maybe just give it a set
    # of nights where it can execute for now.
    basis_functions = []
    # These are crude hard limits. Nominally we would try to
    # pre-schedule these when they would be at the best airamss
    # in the night.
    basis_functions.append(bf.HourAngleLimitBasisFunction(RA=RA, ha_limits=[[20, 24], [0, 4]]))
    basis_functions.append(bf.NotTwilightBasisFunction())
    # Force it to delay about a day
    basis_functions.append(bf.ForceDelayBasisFunction(days_delay=0.8, scheduler_note=scheduler_note))
    # Force it to be in a given observing season
    basis_functions.append(bf.InTimeWindowBasisFunction(mjd_windows=field_info["seasons_on"]))
    basis_functions.append(bf.MoonDistPointRangeBasisFunction(RA, dec, moon_limit=moon_limit))
    basis_functions.append(bf.AirmassPointRangeBasisFunction(RA, dec, nside=nside))
    basis_functions.append(bf.AvoidDirectWind(wind_speed_maximum=wind_speed_maximum, nside=nside))

    # Add a dither detailer, so it dithers between each set
    # of exposures I guess?
    details = []
    details.append(detailers.DitherDetailer(max_dither=0.5, seed=42, per_night=True))
    details.append(detailers.CameraRotDetailer(min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit))

    survey = DeepDrillingSurvey(
        basis_functions,
        RA=RA,
        dec=dec,
        sequence="giriz",
        nvis=[1, 1, 1, 1, 1],
        exptime=exptime,
        nexp=nexp,
        survey_name=scheduler_note,
        detailers=details,
        science_program=science_program,
    )
    return survey


def gen_roman_off_season(
    nside=DEFAULT_NSIDE,
    camera_ddf_rot_limit=75.0,
    exptime=30.0,
    nexp=2,
    wind_speed_maximum=20.0,
    moon_limit=30.0,
    science_program=None,
):
    """Generate a ddf-like survey object to observe the roman
    field every ~3 days in the off-season"""

    warnings.warn("Generating Roman survey place holder. Should probably not be in production.")

    field_info = roman_info()
    RA = field_info["RA"]
    dec = field_info["dec"]

    scheduler_note = "DD: RGES_offseason"

    # Add some feasability basis functions. Maybe just give it a
    # set of nights where it can execute for now.
    basis_functions = []
    # These are crude hard limits. Nominally we would try
    # to pre-schedule these when they would be at the best
    # airamss in the night.
    basis_functions.append(bf.HourAngleLimitBasisFunction(RA=RA, ha_limits=[[20, 24], [0, 4]]))
    basis_functions.append(bf.NotTwilightBasisFunction())
    # Force it to not go every day
    basis_functions.append(bf.ForceDelayBasisFunction(days_delay=3.0, scheduler_note=scheduler_note))
    # Force it to be in a given observing season
    basis_functions.append(bf.InTimeWindowBasisFunction(mjd_windows=field_info["seasons_off"]))
    basis_functions.append(bf.MoonDistPointRangeBasisFunction(RA, dec, moon_limit=moon_limit))
    basis_functions.append(bf.AirmassPointRangeBasisFunction(RA, dec, nside=nside))
    basis_functions.append(bf.AvoidDirectWind(wind_speed_maximum=wind_speed_maximum, nside=nside))

    # Add a dither detailer, so it dithers between each
    # set of exposures I guess?
    details = []
    details.append(detailers.DitherDetailer(max_dither=0.5, seed=42, per_night=True))
    details.append(detailers.CameraRotDetailer(min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit))

    survey = DeepDrillingSurvey(
        basis_functions,
        RA=RA,
        dec=dec,
        sequence="griz",
        nvis=[1, 1, 1, 1],
        exptime=exptime,
        nexp=nexp,
        survey_name=scheduler_note,
        detailers=details,
        science_program=science_program,
    )
    return survey
