__all__ = ("define_ddf_seq", "gen_ddf_surveys")

import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import rubin_scheduler.scheduler.detailers as detailers
import rubin_scheduler.scheduler.example.lsst_ddf_presched as ddf_presched
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.example.lsst_surveys import (
    EXPTIME,
    NEXP,
    SCIENCE_PROGRAM,
    U_EXPTIME,
    U_NEXP,
    safety_masks,
)
from rubin_scheduler.scheduler.surveys import ScriptedSurvey
from rubin_scheduler.scheduler.utils import ScheduledObservationArray
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD

logger = logging.getLogger(__name__)


def define_ddf_seq() -> pd.DataFrame:
    """Define the sequences for each field"""

    short_squences = [
        {
            "u": 3,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 33,
        },
        {
            "y": 2,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 33,
        },
        {
            "g": 2,
            "i": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 56,
            "even_odd": "even",
        },
        {
            "r": 2,
            "z": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 56,
            "even_odd": "odd",
        },
    ]

    shallow_squences = [
        {
            "u": 3,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "y": 2,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "g": 2,
            "i": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 100,
            "even_odd": "even",
        },
        {
            "r": 2,
            "z": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 100,
            "even_odd": "odd",
        },
    ]

    deep_sequences = [
        {
            "u": 8,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "y": 20,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "g": 2,
            "i": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 200,
            "even_odd": "even",
        },
        {
            "r": 2,
            "z": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 200,
            "even_odd": "odd",
        },
        {
            "g": 4,
            "r": 18,
            "i": 55,
            "z": 52,
            "season_length": 180,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 110,
        },
    ]

    euclid_deep_seq = [
        {
            "u": 30,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "y": 40,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "g": 4,
            "i": 4,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 200,
            "even_odd": "even",
        },
        {
            "r": 4,
            "z": 4,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 200,
            "even_odd": "odd",
        },
        {
            "g": 8,
            "r": 36,
            "i": 110,
            "z": 104,
            "season_length": 125,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
    ]

    short_seasons = {
        "XMM_LSS": [0, 10],
        "ELAISS1": [0, 10],
        "ECDFS": [0, 10],
        "EDFS_a": [0, 10],
    }

    shallow_seasons = {
        "COSMOS": [0, 4, 5, 6, 7, 8, 9, 10],
        "XMM_LSS": [1, 2, 3, 5, 6, 7, 8, 9],
        "ELAISS1": [1, 2, 3, 4, 6, 7, 8, 9],
        "ECDFS": [1, 2, 3, 4, 5, 7, 8, 9],
        "EDFS_a": [2, 3, 4, 5, 6, 7, 8, 9],
    }

    deep_seasons = {
        "COSMOS": [1, 2, 3],
        "XMM_LSS": [4],
        "ELAISS1": [5],
        "ECDFS": [6],
        "EDFS_a": [1],
    }

    dataframes = []

    for ddf_name in short_seasons:
        for season in short_seasons[ddf_name]:
            dict_for_df = {
                "ddf_name": ddf_name,
                "season": season,
                "even_odd": "None",
            }
            for key in "ugrizy":
                dict_for_df[key] = 0

            for seq in short_squences:
                row = copy.copy(dict_for_df)
                for key in seq:
                    row[key] = seq[key]
                dataframes.append(pd.DataFrame.from_dict(row, orient="index").T)

    for ddf_name in shallow_seasons:
        for season in shallow_seasons[ddf_name]:
            dict_for_df = {
                "ddf_name": ddf_name,
                "season": season,
                "even_odd": "None",
            }
            for key in "ugrizy":
                dict_for_df[key] = 0

            for seq in shallow_squences:
                row = copy.copy(dict_for_df)
                for key in seq:
                    row[key] = seq[key]
                dataframes.append(pd.DataFrame.from_dict(row, orient="index").T)

    for ddf_name in deep_seasons:
        for season in deep_seasons[ddf_name]:
            dict_for_df = {
                "ddf_name": ddf_name,
                "season": season,
                "even_odd": "None",
            }
            for key in "ugrizy":
                dict_for_df[key] = 0
            if ddf_name == "EDFS_a":
                for seq in euclid_deep_seq:
                    row = copy.copy(dict_for_df)
                    for key in seq:
                        row[key] = seq[key]
                    dataframes.append(pd.DataFrame.from_dict(row, orient="index").T)
            else:
                for seq in deep_sequences:
                    row = copy.copy(dict_for_df)
                    for key in seq:
                        row[key] = seq[key]

                    dataframes.append(pd.DataFrame.from_dict(row, orient="index").T)

    result = pd.concat(dataframes)

    return result


def gen_ddf_surveys(
    detailer_list: list[detailers.BaseDetailer] | None = None,
    nside: int = DEFAULT_NSIDE,
    expt: dict | None = None,
    nexp: dict | None = None,
    survey_start: float = SURVEY_START_MJD,
    survey_length: int = 10,
    survey_name: str = "deep drilling",
    science_program: str = SCIENCE_PROGRAM,
    shadow_minutes: float = 30,
    save: bool = True,
    save_filename: str = "example_ddf_array.npz",
    save_path: str = None,
    safety_mask_params: dict | None = None,
) -> list[ScriptedSurvey]:
    """Generate surveys for DDF observations.

    Parameters
    ----------
    detailer_list : `list` [ `rubin_scheduler.scheduler.Detailer` ]
        Detailers for DDFs. Default None.
    nside : `int`
        Nside for the survey. Used for mask basis functions.
    expt : `dict`  { `str` : `float` } or None
        Exposure time for DDF visits.
        Default of None uses defaults of EXPTIME/U_EXPTIME.
    nexp : `dict` { `str` : `int` } or None
        Number of exposures per visit.
        Default of None uses defaults of NEXP/U_NEXP.
    survey_start : `float`
        Start MJD of the survey. Used for prescheduling DDF visits.
    survey_length : `float`
        Length of the survey. Used for prescheduling DDF visits.
        In years.
    science_program : `str`
        Name of the science program for the Survey.
    shadow_minutes : `float`
        The expected length of time for a series of DDF visits
        to execute. Masks area on the sky which will enter unobservable
        regions within shadow_minutes.
    save : `bool`
        Save the resulting ddf array for faster restore next time run.
    save_filename : `str`
        Filename of the saved ddf array.
    save_path : `str`
        Path to saved DDF file. If none, uses get_data_dir to look for it.
    safety_mask_params : `dict` or None
        A dictionary of additional kwargs to ass to the standard safety masks.

    Returns
    -------
    ddf_surveys : `list` [ `ScriptedSurvey` ]
        A list of Scripted surveys configured with pre-scheduled DDF visits.
    """
    if safety_mask_params is None:
        safety_mask_params = {}
        safety_mask_params["nside"] = nside
    else:
        safety_mask_params = copy.deepcopy(safety_mask_params)
    if "shadow_minutes" not in safety_mask_params or safety_mask_params["shadow_minutes"] < shadow_minutes:
        safety_mask_params["shadow_minutes"] = shadow_minutes

    if expt is None:
        expt = {
            "u": U_EXPTIME,
            "g": EXPTIME,
            "r": EXPTIME,
            "i": EXPTIME,
            "z": EXPTIME,
            "y": EXPTIME,
        }
    if nexp is None:
        nexp = {"u": U_NEXP, "g": NEXP, "r": NEXP, "i": NEXP, "z": NEXP, "y": NEXP}

    # Potential pre-computed obs_array:
    if save_path is None:
        save_path = Path(get_data_dir(), "scheduler")
    # Potetial pre-computed obs_array:
    pre_comp_file = Path(save_path, save_filename)

    # Hash of the files that define the DDF sequences, to identify
    # if the saved file comes from the same sources.
    hash_digest = ddf_presched.calculate_checksum([__file__, ddf_presched.__file__])
    passed_kwargs = {
        "expt": expt,
        "nexp": nexp,
        "survey_start": survey_start,
        "survey_length": survey_length,
        "science_program": science_program,
    }

    # Always try to load the pre-computed data.
    obs_array = None
    if os.path.exists(pre_comp_file):
        loaded = np.load(pre_comp_file, allow_pickle=True)
        if loaded["hash_digest"] == hash_digest:
            # Check that all the kwargs match
            kwargs_match = True
            for key in passed_kwargs:
                if passed_kwargs[key] != loaded[key]:
                    kwargs_match = False

            if kwargs_match:
                logger.info("Loading DDF array from %s" % pre_comp_file)
                obs_array_loaded = loaded["obs_array"]
                # Convert back to a full ScheduledObservationArray?
                obs_array = ScheduledObservationArray(obs_array_loaded.size)
                for key in obs_array_loaded.dtype.names:
                    obs_array[key] = obs_array_loaded[key]
        loaded.close()

    if obs_array is None:
        logger.info("Generating DDF array")
        ddf_dataframe = define_ddf_seq()

        obs_array = ddf_presched.generate_ddf_scheduled_obs(
            ddf_dataframe,
            expt=expt,
            nsnaps=nexp,
            survey_start_mjd=survey_start,
            survey_length=survey_length,
            science_program=science_program,
        )
        # Save computation for later
        if save:
            logger.info("Saving DDF array to %s" % pre_comp_file)
            np.savez(
                pre_comp_file,
                obs_array=obs_array.view(np.ndarray),
                hash_digest=hash_digest,
                expt=expt,
                nexp=nexp,
                survey_start=survey_start,
                survey_length=survey_length,
                science_program=science_program,
            )

    survey1 = ScriptedSurvey(
        safety_masks(**safety_mask_params),
        nside=nside,
        detailers=detailer_list,
        survey_name=survey_name,
        before_twi_check=False,
    )
    survey1.set_script(obs_array)

    return [survey1]
