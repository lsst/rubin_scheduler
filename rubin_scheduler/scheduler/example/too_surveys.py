__all__ = ("gen_too_surveys",)

from copy import deepcopy

import numpy as np
import numpy.typing as npt

import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.detailers import BandPickToODetailer
from rubin_scheduler.scheduler.surveys import ToOScriptedSurvey
from rubin_scheduler.utils import DEFAULT_NSIDE

from .lsst_surveys import EXPTIME, NEXP, SCIENCE_PROGRAM, safety_masks


def gen_too_surveys(
    nside: int = DEFAULT_NSIDE,
    detailer_list: list[detailers.BaseDetailer] | None = None,
    too_footprint: npt.NDArray | None = None,
    long_exp_nsnaps: int = 2,
    n_snaps: int = NEXP,
    science_program: str = SCIENCE_PROGRAM,
    safety_mask_params: dict | None = None,
) -> list[ToOScriptedSurvey]:
    """Generate a list of ToO surveys to follow up
    events passed in Conditions.

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use. Default to DEFAULT_NSIDE.
    detailer_list : `list` of `detailers.BaseDetailer`
        List of survey detailers.
    too_footprint : `np.ndarray` or None
        Footprint to contain ToOs within (such as the lsst footprint).
    long_exp_nsnaps : `int`
        The number of snaps for longer exposures. (60s??)
    n_snaps : `int`
        The number of snaps per visit for other exposures. (??)
    science_program : `str`
        Metadata to identify the science program for the visit.
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to mass to the standard safety masks.

    Returns
    -------
    too_surveys : `list` [ `ToOScriptedSurvey` ]
        A list of ToO surveys configured to trigger a pre-specified sequence
        of visits in response to ToO events in the Conditions objects.
    """
    if safety_mask_params is None:
        safety_mask_params = {}
        safety_mask_params["nside"] = nside
    else:
        safety_mask_params = deepcopy(safety_mask_params)
    # No value of shadow_minutes with ToO surveys?
    masks = safety_masks(**safety_mask_params)

    too_surveys = []

    ############
    # Generic GW followup
    ############

    # XXX---There's some extra stuff about do different things
    # if there's limited time before it sets. Let's see if this can
    # work first

    # XXX--instructions say do 4th night only 1/3 of the time.
    # Just leaving off for now

    times = np.array([0, 24, 48], float)
    bands_at_times = ["ugrizy", "ugrizy", "ugrizy"]
    nvis = [3, 1, 1]
    exptimes = [120.0, 120.0, 120.0]
    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_A"],
            survey_name="ToO, GW_case_A",
            flushtime=48.0,
            n_snaps=long_exp_nsnaps,
            # Update target_name to match the alert event ID
            target_name_base="GW_case_A",
            observation_reason="too_gw_case_a",
            science_program=science_program,
        )
    )

    ############
    # GW gold and GW unidentified gold
    ############

    times = np.array([0, 2, 4, 24, 48, 72], float)
    bands_at_times = ["gri", "gri", "gri", "ri", "ri", "ri"]
    nvis = [4, 4, 4, 6, 6, 6]
    exptimes = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_B", "GW_case_C"],
            survey_name="ToO, GW_case_B_C",
            # Update target_name to match the alert event ID
            target_name_base="GW_case_B_C",
            observation_reason="too_gw_case_b_c",
            science_program=science_program,
            flushtime=48,
            n_snaps=long_exp_nsnaps,
        )
    )

    ############
    # GW silver and GW unidentified silver
    ############

    times = np.array([0, 24, 48, 72], float)
    bands_at_times = ["gi", "gi", "gi", "gi"]
    nvis = [1, 4, 4, 4]
    exptimes = [30.0, 30.0, 30.0, 30.0]
    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_D", "GW_case_E"],
            survey_name="ToO, GW_case_D_E",
            # Update target_name to match the alert event ID
            target_name_base="GW_case_D_E",
            observation_reason="too_gw_case_d_e",
            science_program=science_program,
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
    times = np.array([0, 2, 7, 9, 39], float) * 24
    bands_at_times = ["rzi"] * times.size
    nvis = [1] * times.size
    exptimes = [EXPTIME] * times.size

    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["BBH_case_A", "BBH_case_B", "BBH_case_C"],
            survey_name="ToO, BBH",
            # Update target_name to match the alert event ID
            target_name_base="BBH",
            observation_reason="too_bbh",
            science_program=science_program,
            flushtime=48,
            n_snaps=n_snaps,
            event_gen_detailers=event_detailers,
        )
    )

    ############
    # Lensed BNS
    ############

    times = np.array([1.0, 1.0, 25, 25, 49, 49], float)
    bands_at_times = ["g", "r"] * 3
    nvis = [1, 3] * 3
    exptimes = [EXPTIME, EXPTIME] * 3

    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["lensed_BNS_case_A"],
            survey_name="ToO, LensedBNS_A",
            # Update target_name to match the alert event ID
            target_name_base="LensedBNS_A",
            observation_reason="too_lensed_bns_a",
            science_program=science_program,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    # This is the small skymap (15 deg^2 case)
    times = np.array([1.0, 1.0, 25, 25, 49, 49], float)
    bands_at_times = ["g", "r"] * 3
    nvis = [180, 120] * 3
    exptimes = [30] * times.size

    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["lensed_BNS_case_B"],
            survey_name="ToO, LensedBNS_B",
            # Update target_name to match the alert event ID
            target_name_base="LensedBNS_B",
            observation_reason="too_lensed_bns_b",
            science_program=science_program,
            flushtime=48.0,
            n_snaps=long_exp_nsnaps,
        )
    )

    ############
    # Neutrino detector followup
    ############

    times = np.array([0.0, 0.0, 15.0 / 60.0, 0.0, 24.0, 24.0, 144.0, 144.0], float)
    bands_at_times = ["u", "g", "r", "z", "g", "r", "g", "rz"]
    exptimes = [
        30,
        30,
        EXPTIME,
        EXPTIME,
        30,
        EXPTIME,
        30,
        EXPTIME,
    ]
    nvis = [1, 4, 1, 1, 4, 1, 4, 1]

    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["neutrino"],
            survey_name="ToO, neutrino",
            # Update target_name to match the alert event ID
            target_name_base="neutrino",
            observation_reason="too_neutrino",
            science_program=science_program,
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

    times = np.array([0, 33 / 60.0, 66 / 60.0], float)
    bands_at_times = ["r"] * 3
    nvis = [1] * 3
    exptimes = [EXPTIME] * 3

    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SSO_night"],
            survey_name="ToO, SSO_night",
            # Update target_name to match the alert event ID
            target_name_base="SSO_night",
            observation_reason="too_sso_general",
            science_program=science_program,
            flushtime=3.0,
            n_snaps=n_snaps,
        )
    )

    times = np.array([0, 10 / 60.0, 20 / 60.0])
    bands_at_times = ["z"] * 3
    nvis = [2] * 3
    exptimes = [15.0] * 3

    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SSO_twilight"],
            survey_name="ToO, SSO_twi",
            # Update target_name to match the alert event ID
            target_name_base="SSO_twi",
            observation_reason="too_sso_twi",
            science_program=science_program,
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

    times = np.array([0, 0, 0, 0] * 4, float)
    bands_at_times = ["i", "i", "i", "i"] * 4
    nvis = [1, 1, 1, 1] * 4
    exptimes = [1, 15, 1, 15] * 4

    too_surveys.append(
        ToOScriptedSurvey(
            masks,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SN_Galactic"],
            survey_name="ToO, galactic SN",
            # Update target_name to match the alert event ID
            target_name_base="SN_Galactic",
            observation_reason="too_sn_galactic",
            science_program=science_program,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    return too_surveys
