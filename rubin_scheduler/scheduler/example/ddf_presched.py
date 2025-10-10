__all__ = (
    "generate_ddf_scheduled_obs",
    "ddf_slopes",
    "match_cumulative",
    "optimize_ddf_times",
)

import os
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.utils import ScheduledObservationArray
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import SURVEY_START_MJD, calc_season, ddf_locations


def ddf_slopes(
    ddf_name: str,
    raw_obs: npt.NDArray,
    night_season: npt.NDArray,
    season_seq: int = 30,
    min_season_length: float = 0,
    boost_early_factor: float | None = None,
    boost_factor_third: float | None = None,
    boost_factor_fractional: float | None = None,
) -> npt.NDArray:
    """
    Let's make custom slopes for each DDF

    Parameters
    ----------
    ddf_name : `str`
       The DDF name to use
    raw_obs : `np.array`, (N,)
        An array with values of 1 or zero. One element per night, value of
        1 indicates the night is during an active observing season.
    night_season : `np.array`, (N,)
        An array of floats with the fractional season value.
        Season values range from 0-1 for the first season;
        0.5 would be the "peak" of seasonal visibility.
        These should ONLY be the night the field should be 'active'
        (take out the season ends first).
    season_seq : `int`, optional
        Number of sequences to try to place into each season.
        Default of 30.
    min_season_length : `float`, optional
        The minimum season length in days to allow before discarding the
        season. Default of 0 currently doesn't discard any seasons,
        on the basis that most of the short season concerns will be
        early in the survey and we'd rather have some observations early
        than wait.
    boost_early_factor : `float`, optional
        Number of early seasons to give a boost to
    boost_factor_third : `float`
        If boosting, the boot factor to use on the third season.
    boost_factor_fractional : `float`
        If there is an initial partial season, also boost that one by
        the given factor.
    """

    int_season = np.floor(night_season)

    n_season = night_season[np.where(raw_obs)]
    r_season = int_season[np.where(raw_obs)]
    season_list = np.unique(r_season)

    # Calculate season length for each season
    # In general this should be the same each season except for first and last
    season_length = np.zeros(len(season_list), float)
    for i, season in enumerate(season_list):
        match = np.where(r_season == season)
        season_length[i] = n_season[match].max() - n_season[match].min()

    # Determine goal number of sequences in each season.
    season_vals = np.ones(len(season_list), float) * season_seq
    # Adjust other seasons, relative to the max season length.
    season_vals = season_vals * season_length / np.max(season_length)
    # EXCEPT - throw out seasons which are too short
    too_short = np.where(season_length < (min_season_length / 365.25))
    season_vals[too_short] = 0

    # Add extra adjustment to boost visits in seasons 0, 1, and 2
    # and maybe -1.
    if boost_early_factor is not None:
        early_season = np.where(season_list == -1)[0]
        if (len(early_season) > 0) & (boost_factor_fractional > 0):
            season_vals[early_season] = season_seq * boost_factor_fractional
        first_full_two_seasons = np.where((season_list == 0) | (season_list == 1))
        season_vals[first_full_two_seasons] *= boost_early_factor
        third_season = np.where(season_list == 2)
        season_vals[third_season] *= boost_factor_third

    # Round the season_vals -- we're looking for integer numbers of sequences
    season_vals = np.round(season_vals)

    # assign cumulative values
    cumulative_desired = np.zeros(raw_obs.size, dtype=float)
    for i, season in enumerate(season_list):
        this_season = np.where(int_season == season)
        # raw_obs gives a way to scale the season_vals across the nights
        # in the season -- raw_obs = 1 for peak of season, 0 when
        # beyond the season_unobs_frac, and a fraction for low_season_frac
        cumulative = np.cumsum(raw_obs[this_season])
        if cumulative.max() > 0:
            cumulative = cumulative / cumulative.max() * season_vals[i]
            cumulative_desired[this_season] = cumulative + np.max(cumulative_desired)

    return cumulative_desired


def match_cumulative(
    cumulative_desired: npt.NDArray,
    mask: npt.NDArray | None = None,
    no_duplicate: bool = True,
) -> npt.NDArray:
    """Generate a schedule that tries to match the desired cumulative
    distribution given a mask

    Parameters
    ----------
    cumulative_desired : `np.array`, float
        An array with the cumulative number of desired observations.
        Elements  are assumed to be evenly spaced.
    mask : `np.array`, bool or int (None)
        Set to zero for indices that cannot be scheduled
    no_duplicate : `bool` (True)
        If True, only 1 event can be scheduled per element

    Returns
    -------
    schedule : `np.array`
        The resulting schedule, with values marking number of events
        in that cell.
    """

    rounded_desired = np.round(cumulative_desired)
    sched = cumulative_desired * 0
    if mask is None:
        mask = np.ones(sched.size)

    valid = np.where(mask > 0)[0].tolist()
    x = np.arange(sched.size)

    drd = np.diff(rounded_desired)
    step_points = np.where(drd > 0)[0] + 1

    # would be nice to eliminate this loop, but it's not too bad.
    # can't just use searchsorted on the whole array, because then there
    # can be duplicate values, and array[[n,n]] = 1 means that extra
    # match gets lost.
    for indx in step_points:
        left = np.searchsorted(x[valid], indx)
        right = np.searchsorted(x[valid], indx, side="right")
        d1 = indx - left
        d2 = right - indx
        if d1 < d2:
            sched_at = left
        else:
            sched_at = right

        # If we are off the end
        if sched_at >= len(valid):
            sched_at -= 1

        sched[valid[sched_at]] += 1
        if no_duplicate:
            valid.pop(sched_at)

    return sched


def optimize_ddf_times(
    ddf_name: str,
    ddf_RA: float,
    ddf_grid: float,
    sun_limit: float = -18,
    sequence_time: float = 60.0,
    airmass_limit: float = 2.5,
    sky_limit: float | None = None,
    g_depth_limit: float = 23.5,
    offseason_length: float = 73.05,
    low_season_frac: float = 0,
    low_season_rate: float = 0.3,
    survey_start_mjd: float = SURVEY_START_MJD,
    season_seq: int = 30,
    boost_early_factor: float | None = None,
    boost_factor_third: float | None = None,
    boost_factor_fractional: float | None = None,
    only_season: int | None = None,
    mask_even_odd: bool | None = None,
    moon_illum_lt=None,
    moon_illum_gt=None,
    early_late_season_only: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """

    Parameters
    ----------
    ddf_name : `str`
        The name of the DDF, used to identify visits scheduled for each DDF.
    ddf_RA : `float`
        The RA of the DDF, used to calculate the season values. In degrees.
    ddf_grid : `np.array`
        An array with info for the DDFs. Generated by the
        rubin_scheduler.scheduler/surveys/generate_ddf_grid.py` script
        The time spacing in this is approximately 15 or 30 minutes,
        and includes visits which may be during twilight.
    sun_limit : `float`, optional
        The maximum sun altitude allowed when prescheduling DDF visits.
        In degrees.
    season_seq : `int`
        Number of sequences to try to place into each season.
        Default of 30.
    sequence_time : `float`, optional
        Expected time for each DDF sequence, used to avoid hitting the
        sun_limit (running DDF visits into twilight). In minutes.
    airmass_limit : `float`, optional
        The maximum airmass allowed when prescheduling DDF visits.
    sky_limit : `float`, optional
        The maximum skybrightness allowed when prescheduling DDF visits.
        This is a skybrightness limit in g band (mags).
        Default None imposes no limit.
    g_depth_limit : `float`, optional
        The minimum g band five sigma depth limit allowed when prescheduling
        DDF visits. This is a depth limit in g band (mags).
        The depth is calculated using skybrightness from skybrightness_pre,
        a nominal FWHM_500 seeing at zenith of 0.7" (resulting in airmass
        dependent seeing) and exposure time.
        Default 23.5. Set to None for no limit.
    offseason_length : `float`, optional
        Number of days to have as the unobservable off-season. Observations
        are observed when offseason_length < night < 365.25 - offseason_length
        counting night=0 as when the sun is at the RA of the DDF. Default of
        73.05 days results in observing seasons of 292.2 days.
    low_season_frac : `float`, optional
        Defines the end of the range of the "low cadence" prescheduled
        observing season.
        The "standard cadence" season runs from:
        low_season_frac < season < (1 - low_season_frac)
        For an 'accordian' style DDF with fewer observations near
        the ends of the season, set this to a value larger than
        `season_unobs_frac`. Values smaller than `season_unobs_frac`
        will result in DDFs with a constant rate throughout the season.
    low_season_rate : `float`, optional
        Defines the rate to use within the low cadence portion
        of the season. During the standard season, the 'rate' is 1.
        This is used in `ddf_slopes` to define the desired number of
        cumulative observations for each DDF over time.
    survey_start_mjd : `float`, optional
        The MJD of the start of the survey. Used to identify the
        starting point when counting seasons.
        Default SURVEY_START_MJD.
    only_season : `int`
        Only optimize for a single season. Default None.
    mask_even_odd : `bool`
        If True, mask even nights, if False, mask odd. Default
        None masks nothing.
    early_late_season_only : `bool`
        Only schedule things in the early and late season.
    """
    # Convert to fraction for convienence
    season_unobs_frac = offseason_length / 365.25

    # Convert sun_limit and sequence_time to values expected internally.
    sun_limit = np.radians(sun_limit)
    sequence_time = sequence_time / 60.0 / 24.0  # to days

    # Calculate the night value for each grid point.
    almanac = Almanac(mjd_start=ddf_grid["mjd"].min())
    almanac_indx = almanac.mjd_indx(ddf_grid["mjd"])
    night = almanac.sunsets["night"][almanac_indx]

    ngrid = ddf_grid["mjd"].size

    # Set the sun mask.
    sun_mask = np.ones(ngrid, dtype=int)
    sun_mask[np.where(ddf_grid["sun_alt"] >= sun_limit)] = 0

    # Set the airmass mask.
    airmass_mask = np.ones(ngrid, dtype=int)
    airmass_mask[np.where(ddf_grid["%s_airmass" % ddf_name] >= airmass_limit)] = 0

    # Set the sky_limit mask, if provided.
    sky_mask = np.ones(ngrid, dtype=int)
    if sky_limit is not None:
        sky_mask[np.where(ddf_grid["%s_sky_g" % ddf_name] <= sky_limit)] = 0
        sky_mask[np.where(np.isnan(ddf_grid["%s_sky_g" % ddf_name]))] = 0

    # Set the m5 / g_depth_limit mask if provided.
    m5_mask = np.zeros(ngrid, dtype=bool)
    m5_mask[np.isfinite(ddf_grid["%s_m5_g" % ddf_name])] = 1
    if g_depth_limit is not None:
        m5_mask[np.where(ddf_grid["%s_m5_g" % ddf_name] < g_depth_limit)] = 0

    eo_mask = 1
    if mask_even_odd is not None:
        eo_mask = np.ones(ngrid, dtype=bool)
        is_odd_indx = night % 2 != 0
        is_even_indx = night % 2 == 0
        if mask_even_odd:
            eo_mask[is_even_indx] = 0
        else:
            eo_mask[is_odd_indx] = 0

    moon_illum_mask = 1
    if moon_illum_lt is not None:
        moon_illum_mask = np.ones(ngrid, dtype=bool)
        indx = np.where(ddf_grid["moon_phase"] > moon_illum_lt)[0]
        moon_illum_mask[indx] = 0
    if moon_illum_gt is not None:
        moon_illum_mask = np.ones(ngrid, dtype=bool)
        indx = np.where(ddf_grid["moon_phase"] < moon_illum_gt)[0]
        moon_illum_mask[indx] = 0

    # Combine the masks.
    big_mask = sun_mask * airmass_mask * sky_mask * m5_mask * eo_mask * moon_illum_mask

    # Do we need to grow the mask to make sure we don't run into a masked
    # time while executing the sequence?
    mask_timestep = ddf_grid["mjd"][1] - ddf_grid["mjd"][0]
    n_mask_steps = int(np.ceil(sequence_time / mask_timestep))
    init_indx = np.where(big_mask == 0)[0]
    for offset in range(1, n_mask_steps + 1):
        indx = init_indx - offset
        indx = indx[indx >= 0]
        big_mask[indx] = 0

    # Identify which nights are useful to preschedule DDF visits.
    potential_nights = np.unique(night[np.where(big_mask > 0)])
    # prevent a repeat sequence in a night
    unights, indx = np.unique(night, return_index=True)
    night_mjd = ddf_grid["mjd"][indx]

    # Calculate season values for each night.
    night_season = calc_season(ddf_RA, night_mjd, survey_start_mjd)

    # convert so we start at season 0 no matter what.
    night_season = night_season - np.floor(np.min(night_season))

    # only making a fit for a single season.
    if only_season is not None:
        indx = np.where(np.floor(night_season).astype(int) == int(only_season))[0]
        night_season = night_season[indx]
        unights = unights[indx]
        night_mjd = night_mjd[indx]

    # Mod by 1 to turn the season value in each night a simple 0-1 value
    season_mod = night_season % 1
    # Remove the portions of the season beyond season_unobs_frac.
    # Note that the "season" does include times where the field wouldn't
    # really have been observable (depending on declination).
    # Small values of season_unobs_frac may not influence the usable season.

    out_season = np.where((season_mod < season_unobs_frac) | (season_mod > (1.0 - season_unobs_frac)))
    # Identify the low-season times.
    low_season = np.where((season_mod < low_season_frac) | (season_mod > (1.0 - low_season_frac)))

    # Turn these into the 'rate scale' per night that goes to `ddf_slopes`
    if early_late_season_only:
        raw_obs = np.zeros(unights.size)
        raw_obs[low_season] = low_season_rate
    else:
        raw_obs = np.ones(unights.size)
        raw_obs[out_season] = 0
        raw_obs[low_season] = low_season_rate

    # Can fail on a partial season with nothing in bounds
    if np.sum(raw_obs) > 0:
        cumulative_desired = ddf_slopes(
            ddf_name,
            raw_obs,
            night_season,
            season_seq=season_seq,
            boost_early_factor=boost_early_factor,
            boost_factor_third=boost_factor_third,
            boost_factor_fractional=boost_factor_fractional,
        )
    else:
        return [], [], [], []

    # Identify which nights (only scheduling 1 sequence per night)
    # would be usable, based on the masks above.
    night_mask = unights * 0
    # which unights are in potential nights
    indx = np.isin(unights, potential_nights)
    night_mask[indx] = 1

    # scale things down if we don't have enough nights
    n_possible_nights = np.sum(night_mask)
    if n_possible_nights < np.max(cumulative_desired):
        warnings.warn(
            "Asked for %i %s sequences, but only %i nights. Decreasing sequences."
            % (np.max(cumulative_desired), ddf_name, n_possible_nights)
        )
        cumulative_desired = cumulative_desired / np.max(cumulative_desired) * n_possible_nights

    # Calculate expected cumulative sequence schedule.
    unight_sched = match_cumulative(cumulative_desired, mask=night_mask)
    cumulative_sched = np.cumsum(unight_sched)

    nights_to_use = unights[np.where(unight_sched == 1)]

    # For each night, find the best time in the night and preschedule the DDF.
    # XXX--probably need to expand this part to resolve the times when
    # multiple things get scheduled

    mjds = []
    for night_check in nights_to_use:
        in_night = np.where(
            (night == night_check) & (np.isfinite(ddf_grid["%s_m5_g" % ddf_name])) & (big_mask > 0)
        )[0]
        m5s = ddf_grid["%s_m5_g" % ddf_name][in_night]
        # we could intorpolate this to get even better than 15 min
        # resolution on when to observe
        max_indx = np.where(m5s == m5s.max())[0].min()
        mjds.append(ddf_grid["mjd"][in_night[max_indx]])
    return mjds, night_mjd, cumulative_desired, cumulative_sched


def generate_ddf_scheduled_obs(
    configs_df: pd.DataFrame,
    data_file: str | None = None,
    mjd_tol: float = 15.0,
    expt: dict = {"u": 38.0, "g": 29.2, "r": 29.2, "i": 29.2, "z": 29.2, "y": 29.2},
    alt_min: float = 25.0,
    alt_max: float = 85.0,
    HA_min: float = 21.0,
    HA_max: float = 3.0,
    sun_alt_max: float = -18.0,
    moon_min_distance: float = 25.0,
    dist_tol: float = 3.0,
    bands: str = "ugrizy",
    nsnaps: dict = {"u": 1, "g": 2, "r": 2, "i": 2, "z": 2, "y": 2},
    survey_start_mjd: float = SURVEY_START_MJD,
    survey_length: float = 10.0,
    low_season_frac: float = 0.0,
    low_season_rate: float = 0.3,
    overhead: float = 2.0,
    illum_limit: float = 40.0,
    science_program: str | None = None,
) -> ScheduledObservationArray:
    """

    Parameters
    ----------
    configs_df : `pd.DataFrame`
        A DataFrame containing rows of DDF sequences to schedule
    data_file : `path` (None)
        The data file to use for DDF airmass, m5, etc. Defaults to
        using whatever is in rubin_sim_data/scheduler directory.
    flush_length : `float` (2)
        How long to keep a scheduled observation around before it is
        considered failed and flushed (days).
    mjd_tol : `float` (15)
        How close an observation must be in time to be considered
        matching a scheduled observation (minutes).
    expt : `float` (30)
        Total exposure time per visit (seconds).
    alt_min/max : `float` (25, 85)
        The minimum and maximum altitudes to permit observations to
        happen (degrees).
    HA_min/max : `float` (21, 3)
        The hour angle limits to permit observations to happen (hours).
    moon_min_distance : `float`
        The minimum distance to demand from the moon (degrees).
    dist_tol : `float` (3)
        The distance tolerance for a visit to be considered matching a
        scheduled observation (degrees).
    nvis_master : list of ints ([8, 10, 20, 20, 24, 18])
        The number of visits to make per band
    bands : `str` (ugrizy)
        The band names.
    nsnaps : `list of ints` ([1, 2, 2, 2, 2, 2])
        The number of snaps to use per band
    survey_start_mjd : `float`
        Starting MJD of the survey. Default None, which calls
        rubin_sim.utils.SURVEY_START_MJD
    survey_length : `float`
        Length of survey (years). Default 10.
    sequence_time : `float`, optional
        Expected time for each DDF sequence, used to avoid hitting the
        sun_limit (running DDF visits into twilight). In minutes.
    season_unobs_frac : `float`, optional
        Defines the end of the range of the prescheduled observing season.
        season runs from 0 (sun's apparent position is at the RA of the DDF)
        to 1 (sun returns to an apparent position in the RA of the DDF).
        The scheduled season runs from:
        season_unobs_frac < season < (1-season_unobs_fract)
    low_season_frac : `float`, optional
        Defines the end of the range of the "low cadence" prescheduled
        observing season.
        The "standard cadence" season runs from:
        low_season_frac < season < (1 - low_season_frac)
        For an 'accordian' style DDF with fewer observations near
        the ends of the season, set this to a value larger than
        `season_unobs_frac`. Values smaller than `season_unobs_frac`
        will result in DDFs with a constant rate throughout the season.
    low_season_rate : `float`, optional
        Defines the rate to use within the low cadence portion
        of the season. During the standard season, the 'rate' is 1.
        This is used in `ddf_slopes` to define the desired number of
        cumulative observations for each DDF over time.
    illum_limit : `float`
        The moon illumination limit for when u and y are loaded.
        Default 40 (percent).
    """
    if data_file is None:
        data_file = os.path.join(get_data_dir(), "scheduler", "ddf_grid_fine.npz")

    mjd_tol = mjd_tol / 60 / 24.0  # minutes to days
    alt_min = np.radians(alt_min)
    alt_max = np.radians(alt_max)
    dist_tol = np.radians(dist_tol)
    sun_alt_max = np.radians(sun_alt_max)
    moon_min_distance = np.radians(moon_min_distance)

    ddfs = ddf_locations()
    ddf_data = np.load(data_file)
    ddf_grid = ddf_data["ddf_grid"].copy()

    mjd_max = survey_start_mjd + survey_length * 365.25

    # check if our pre-computed grid is over the time range we think
    # we are scheduling for
    if (ddf_grid["mjd"].min() > survey_start_mjd) | (ddf_grid["mjd"].max() < mjd_max):
        warnings.warn("Pre-computed DDF properties don't match requested survey times")

    in_range = np.where((ddf_grid["mjd"] >= survey_start_mjd) & (ddf_grid["mjd"] <= mjd_max))
    ddf_grid = ddf_grid[in_range]

    # can loop over each row to generate the observations that
    # row calls for
    all_scheduled_obs = []
    for index, row in configs_df.iterrows():

        ddf_name = row["ddf_name"]
        offseason_length = (365.0 - row["season_length"]) / 2.0  # stupid factor of 2
        n_sequences = row["n_sequences"]

        sequence_time = 0.0
        sequence_dict = {}
        flush_length = row["flush_length"]

        u_only = False
        y_only = False
        sum_filters = row["u"] + row["g"] + row["r"] + row["i"] + row["z"] + row["y"]
        if sum_filters == row["u"]:
            u_only = True
        elif sum_filters == row["y"]:
            y_only = True

        for bandname in "ugrizy":
            sequence_dict[bandname] = row[bandname]
            sequence_time += (expt[bandname] + overhead * nsnaps[bandname]) * row[bandname]

        mask_even_odd = None
        if row["even_odd"].strip() == "even":
            mask_even_odd = True
        elif row["even_odd"].strip() == "odd":
            mask_even_odd = False

        moon_illum_gt = None
        moon_illum_lt = None
        if u_only:
            moon_illum_lt = illum_limit
        if y_only:
            moon_illum_gt = illum_limit

        # Only doing early and late season observations
        if "early" in row["even_odd"].strip():
            if "p_0" in row["even_odd"].strip():
                mask_even_odd = True
            elif "p_2" in row["even_odd"].strip():
                mask_even_odd = False
            mjds = optimize_ddf_times(
                ddf_name,
                ddfs[ddf_name][0],
                ddf_grid,
                sun_limit=-18,
                sequence_time=sequence_time / 60.0,
                airmass_limit=2.5,
                sky_limit=None,
                g_depth_limit=row["g_depth_limit"],
                offseason_length=offseason_length,
                # XXX--magic number should move to config file
                low_season_frac=80.0 / 200.0,
                low_season_rate=1.0,
                survey_start_mjd=survey_start_mjd,
                season_seq=n_sequences,
                only_season=row["season"],
                mask_even_odd=mask_even_odd,
                moon_illum_lt=moon_illum_lt,
                moon_illum_gt=moon_illum_gt,
                early_late_season_only=True,
            )[0]
        else:
            mjds = optimize_ddf_times(
                ddf_name,
                ddfs[ddf_name][0],
                ddf_grid,
                sun_limit=-18,
                sequence_time=sequence_time / 60.0,
                airmass_limit=2.5,
                sky_limit=None,
                g_depth_limit=row["g_depth_limit"],
                offseason_length=offseason_length,
                low_season_frac=0,
                low_season_rate=0.3,
                survey_start_mjd=survey_start_mjd,
                season_seq=n_sequences,
                only_season=row["season"],
                mask_even_odd=mask_even_odd,
                moon_illum_lt=moon_illum_lt,
                moon_illum_gt=moon_illum_gt,
                early_late_season_only=False,
            )[0]

        for mjd in mjds:
            for bandname in sequence_dict:
                if "EDFS" in ddf_name:
                    obs = ScheduledObservationArray(n=int(np.ceil(sequence_dict[bandname] / 2)))
                    obs["RA"] = np.radians(ddfs[ddf_name][0])
                    obs["dec"] = np.radians(ddfs[ddf_name][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt[bandname]
                    obs["band"] = bandname
                    obs["nexp"] = nsnaps[bandname]
                    obs["scheduler_note"] = "DD:%s" % ddf_name
                    obs["target_name"] = "ddf_" + ddf_name.lower()
                    obs["science_program"] = "DD"
                    obs["observation_reason"] = "ddf_" + ddf_name.lower()

                    obs["mjd_tol"] = mjd_tol
                    obs["dist_tol"] = dist_tol
                    # Need to set something for HA limits
                    obs["HA_min"] = HA_min
                    obs["HA_max"] = HA_max
                    obs["alt_min"] = alt_min
                    obs["alt_max"] = alt_max
                    obs["sun_alt_max"] = sun_alt_max
                    all_scheduled_obs.append(obs)

                    obs = ScheduledObservationArray(n=int(np.ceil(sequence_dict[bandname] / 2)))
                    obs["RA"] = np.radians(ddfs[ddf_name.replace("_a", "_b")][0])
                    obs["dec"] = np.radians(ddfs[ddf_name.replace("_a", "_b")][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt[bandname]
                    obs["band"] = bandname
                    obs["nexp"] = nsnaps[bandname]
                    obs["scheduler_note"] = "DD:%s" % ddf_name.replace("_a", "_b")
                    obs["target_name"] = "ddf_" + ddf_name.replace("_a", "_b").lower()
                    obs["science_program"] = science_program
                    obs["observation_reason"] = "ddf_" + ddf_name.lower()

                    obs["mjd_tol"] = mjd_tol
                    obs["dist_tol"] = dist_tol
                    # Need to set something for HA limits
                    obs["HA_min"] = HA_min
                    obs["HA_max"] = HA_max
                    obs["alt_min"] = alt_min
                    obs["alt_max"] = alt_max
                    obs["sun_alt_max"] = sun_alt_max
                    obs["moon_min_distance"] = moon_min_distance
                    all_scheduled_obs.append(obs)

                else:
                    obs = ScheduledObservationArray(n=sequence_dict[bandname])
                    obs["RA"] = np.radians(ddfs[ddf_name][0])
                    obs["dec"] = np.radians(ddfs[ddf_name][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt[bandname]
                    obs["band"] = bandname
                    obs["nexp"] = nsnaps[bandname]
                    obs["scheduler_note"] = "DD:%s" % ddf_name
                    obs["target_name"] = "ddf_" + ddf_name.lower()
                    obs["science_program"] = science_program
                    obs["observation_reason"] = "ddf_" + ddf_name.lower()

                    obs["mjd_tol"] = mjd_tol
                    obs["dist_tol"] = dist_tol
                    # Need to set something for HA limits
                    obs["HA_min"] = HA_min
                    obs["HA_max"] = HA_max
                    obs["alt_min"] = alt_min
                    obs["alt_max"] = alt_max
                    obs["sun_alt_max"] = sun_alt_max
                    obs["moon_min_distance"] = moon_min_distance
                    all_scheduled_obs.append(obs)

    result = np.concatenate(all_scheduled_obs)
    return result
