__all__ = ("gen_too_surveys",)

from copy import deepcopy

import numpy as np

import rubin_scheduler.scheduler.basis_functions as basis_functions
from rubin_scheduler.scheduler.detailers import BandPickToODetailer
from rubin_scheduler.scheduler.surveys import ToOScriptedSurvey
from rubin_scheduler.utils import DEFAULT_NSIDE

from .generate_surveys import EXPTIME as DEFAULT_EXP_TIME


def gen_too_surveys(
    nside=DEFAULT_NSIDE,
    detailer_list=None,
    too_footprint=None,
    split_long=False,
    long_exp_nsnaps=2,
    n_snaps=2,
    wind_speed_maximum=20.0,
    observation_reason="ToO",
    science_program=None,
):
    result = []
    bf_list = []
    bf_list.append(basis_functions.AvoidDirectWind(wind_speed_maximum=wind_speed_maximum, nside=nside))
    bf_list.append(basis_functions.MoonAvoidanceBasisFunction(moon_distance=30.0))
    ############
    # Generic GW followup
    ############

    # XXX---There's some extra stuff about do different things
    # if there's limited time before it sets. Let's see if this can
    # work first

    # XXX--instructions say do 4th night only 1/3 of the time.
    # Just leaving off for now

    times = [0, 24, 48]
    bands_at_times = ["ugrizy", "ugrizy", "ugrizy"]
    nvis = [3, 1, 1]
    exptimes = [120.0, 120.0, 120.0]
    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_A"],
            survey_name="ToO, GW_case_A",
            split_long=split_long,
            flushtime=48.0,
            n_snaps=long_exp_nsnaps,
            target_name_base="GW_case_A",
            observation_reason=observation_reason,
            science_program=science_program,
        )
    )

    ############
    # GW gold and GW unidentified gold
    ############

    times = [0, 2, 4, 24, 48, 72]
    bands_at_times = ["gri", "gri", "gri", "ri", "ri", "ri"]
    nvis = [4, 4, 4, 6, 6, 6]
    exptimes = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_B", "GW_case_C"],
            survey_name="ToO, GW_case_B_C",
            target_name_base="GW_case_B_C",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48,
            n_snaps=long_exp_nsnaps,
        )
    )

    ############
    # GW silver and GW unidentified silver
    ############

    times = [0, 24, 48, 72]
    bands_at_times = ["gi", "gi", "gi", "gi"]
    nvis = [1, 4, 4, 4]
    exptimes = [30.0, 30.0, 30.0, 30.0]
    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_D", "GW_case_E"],
            survey_name="ToO, GW_case_D_E",
            target_name_base="GW_case_D_E",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48,
            n_snaps=long_exp_nsnaps,
            event_gen_detailers=None,
        )
    )

    ############
    # BBH hole-black hole GW merger
    # If nearby (dist < 2200 Mpc) and dark time, use ugi
    # If distant (dist > 2200 Mpc) and dark time, use rgi
    # If bright time, use rzi
    ############

    # XXX--only considering bright objects now.

    # SM-- adding support for different BBH cases. We should have
    # a discussion about how to differentiate these, and if it is
    # possible to do so through the alert stream.

    event_detailers = [
        BandPickToODetailer(
            band_start="z",
            band_end="g",
            distance_limit=30e10,
            check_mounted=True,
            require_dark=True,
        ),
        BandPickToODetailer(
            band_start="r",
            band_end="u",
            distance_limit=2.2e6,
            check_mounted=True,
            require_dark=True,
        ),
    ]
    times = np.array([0, 2, 7, 9, 39]) * 24
    bands_at_times = ["rzi"] * times.size
    nvis = [1] * times.size
    exptimes = [DEFAULT_EXP_TIME] * times.size

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["BBH_case_A", "BBH_case_B", "BBH_case_C"],
            survey_name="ToO, BBH",
            target_name_base="BBH",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48,
            n_snaps=n_snaps,
            event_gen_detailers=event_detailers,
        )
    )

    ############
    # Lensed BNS
    ############

    times = np.array([1.0, 1.0, 25, 25, 49, 49])
    bands_at_times = ["g", "r"] * 3
    nvis = [1, 3] * 3
    exptimes = [DEFAULT_EXP_TIME, DEFAULT_EXP_TIME] * 3

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["lensed_BNS_case_A"],
            survey_name="ToO, LensedBNS_A",
            target_name_base="LensedBNS_A",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    # This is the small skymap (15 deg^2 case)
    times = np.array([1.0, 1.0, 25, 25, 49, 49])
    bands_at_times = ["g", "r"] * 3
    nvis = [180, 120] * 3
    exptimes = [30] * times.size

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["lensed_BNS_case_B"],
            survey_name="ToO, LensedBNS_B",
            target_name_base="LensedBNS_B",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48.0,
            n_snaps=long_exp_nsnaps,
        )
    )

    ############
    # Neutrino detector followup
    ############

    times = [0, 0, 15 / 60.0, 0, 24, 24, 144, 144]
    bands_at_times = ["u", "g", "r", "z", "g", "r", "g", "rz"]
    exptimes = [
        30,
        30,
        DEFAULT_EXP_TIME,
        DEFAULT_EXP_TIME,
        30,
        DEFAULT_EXP_TIME,
        30,
        DEFAULT_EXP_TIME,
    ]
    nvis = [1, 4, 1, 1, 4, 1, 4, 1]

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["neutrino"],
            survey_name="ToO, neutrino",
            target_name_base="neutrino",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=20 * 24,
            n_snaps=n_snaps,
        )
    )

    ############
    # Solar System
    ############
    # For the solar system objects, probably want a custom survey object,
    # but this should work for now. Want to add a detailer to add a dither
    # position.

    times = [0, 33 / 60.0, 66 / 60.0]
    bands_at_times = ["r"] * 3
    nvis = [1] * 3
    exptimes = [DEFAULT_EXP_TIME] * 3

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SSO_night"],
            survey_name="ToO, SSO_night",
            target_name_base="SSO_night",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=3.0,
            n_snaps=n_snaps,
        )
    )

    times = [0, 10 / 60.0, 20 / 60.0]
    bands_at_times = ["z"] * 3
    nvis = [2] * 3
    exptimes = [15.0] * 3

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SSO_twilight"],
            survey_name="ToO, SSO_twi",
            target_name_base="SSO_twi",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=3.0,
            n_snaps=n_snaps,
        )
    )

    ############
    # Galactic Supernova
    ############
    # For galactic supernovae, we want to tile continuously
    # in the region in 1s and 15s exposures for i band, until
    # a counterpart is identified

    times = [0, 0, 0, 0] * 4
    bands_at_times = ["i", "i", "i", "i"] * 4
    nvis = [1, 1, 1, 1] * 4
    exptimes = [1, 15, 1, 15] * 4

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SN_Galactic"],
            survey_name="ToO, galactic SN",
            target_name_base="SN_Galactic",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    return result
