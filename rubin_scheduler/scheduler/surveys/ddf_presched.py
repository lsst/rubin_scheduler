__all__ = ("generate_ddf_scheduled_obs",)

import os
import warnings

import numpy as np

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.utils import scheduled_observation
from rubin_scheduler.utils import calc_season, ddf_locations, survey_start_mjd


def ddf_slopes(ddf_name, raw_obs, night_season, min_season=60 / 365.25):
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
        0.5 would be the "peak" of seasonal visiblity.
        These should ONLY be the night the field should be 'active'
        (take out the season ends first).
    min_season : `float`, optional
        Minimal season length that is useful.
        In units of "season" (fractions of a year).
    """

    # OK, so 258 sequences is ~1% of the survey
    # so a 25.8 sequences is a 0.1% season
    # COSMOS is going to be 0.7% for 3 years, then 0.175 for the rest.

    ss = 30  # standard season, was 45 .. note this may be affected by season length

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
    season_vals = np.ones(len(season_list), float) * ss
    # Throw out seasons which are too short
    too_short = np.where(season_length < min_season)
    season_vals[too_short] = 0
    # Adjust other seasons, relative to the max season length.
    season_vals = season_vals * season_length / np.max(season_length)

    # Add extra adjustment for COSMOS to boost visits in seasons 0-3
    # Note this means boost in first incomplete season too
    if ddf_name == "COSMOS":
        season_vals[0:3] = season_vals[0:3] * 5
        season_vals[3] = season_vals[3] * 2

    if ddf_name == "EDFS_b":
        # EDFS_b ddf visits are allocated some other way
        season_vals = season_vals * 0

    # Round the season_vals so that we're looking for integer sequences
    season_vals = np.round(season_vals)

    # print(list(zip(season_vals, season_length)))

    # assign cumulative values
    cumulative_desired = np.zeros(raw_obs.size, dtype=float)
    for i, season in enumerate(season_list):
        in_season = np.where(int_season == season)
        cumulative = np.cumsum(raw_obs[in_season])
        if cumulative.max() > 0:
            cumulative = cumulative / cumulative.max() * season_vals[i]
            cumulative_desired[in_season] = cumulative + np.max(cumulative_desired)

    return cumulative_desired


def match_cumulative(cumulative_desired, mask=None, no_duplicate=True):
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
    ddf_name,
    ddf_RA,
    ddf_grid,
    sun_limit=-18,
    sequence_time=60.0,
    airmass_limit=2.5,
    sky_limit=None,
    g_depth_limit=23.5,
    season_unobs_frac=0.1,
):
    """

    Parameters
    ----------
    ddf : `str`
        The name of the DDF
    ddf_grid : `np.array`
        An array with info for the DDFs. Generated by the
        rubin_scheduler.scheduler/surveys/generate_ddf_grid.py` script
    season_unobs_frac : `float`
        7.2 month observing season if season_unobs_frac = 0.2
        (shaves 20% off each end of the full year)
    sequence_time : `float`
        How long a sequence is expected to be (minutes). Used to make
        sure things are not scheduled too close to twilight.
    """
    sun_limit = np.radians(sun_limit)
    sequence_time = sequence_time / 60.0 / 24.0  # to days

    # XXX-- double check that I got this right
    ack = ddf_grid["sun_alt"][0:-1] * ddf_grid["sun_alt"][1:]
    night = np.zeros(ddf_grid.size, dtype=int)
    night[np.where((ddf_grid["sun_alt"][1:] >= 0) & (ack < 0))] += 1
    night = np.cumsum(night)
    ngrid = ddf_grid["mjd"].size

    # set a sun, airmass, sky masks
    sun_mask = np.ones(ngrid, dtype=int)
    sun_mask[np.where(ddf_grid["sun_alt"] >= sun_limit)] = 0

    # expand sun mask backwards by the sequence time.
    n_back = np.ceil(sequence_time / (ddf_grid["mjd"][1] - ddf_grid["mjd"][0])).astype(int)
    shadow_indx = np.where(sun_mask == 0)[0] - n_back
    shadow_indx = shadow_indx[np.where(shadow_indx >= 0)]

    sun_mask[shadow_indx] = 0

    airmass_mask = np.ones(ngrid, dtype=int)
    airmass_mask[np.where(ddf_grid["%s_airmass" % ddf_name] >= airmass_limit)] = 0

    sky_mask = np.ones(ngrid, dtype=int)
    if sky_limit is not None:
        sky_mask[np.where(ddf_grid["%s_sky_g" % ddf_name] <= sky_limit)] = 0
        sky_mask[np.where(np.isnan(ddf_grid["%s_sky_g" % ddf_name]) == True)] = 0

    m5_mask = np.zeros(ngrid, dtype=bool)
    m5_mask[np.isfinite(ddf_grid["%s_m5_g" % ddf_name])] = 1

    if g_depth_limit is not None:
        m5_mask[np.where(ddf_grid["%s_m5_g" % ddf_name] < g_depth_limit)] = 0

    big_mask = sun_mask * airmass_mask * sky_mask * m5_mask

    potential_nights = np.unique(night[np.where(big_mask > 0)])

    # prevent a repeat sequence in a night
    unights, indx = np.unique(night, return_index=True)
    night_mjd = ddf_grid["mjd"][indx]
    # The season of each night
    night_season = calc_season(ddf_RA, night_mjd)

    # Mod by 1 to turn the season value in each night a simple 0-1 value
    season_mod = night_season % 1
    # Remove the tails of the seasons which are within "season_unobs_fract"
    # Note that the "season" does include times where the field wouldn't
    # really have been observable, due to being too close to the sun.
    # Small values of season_unobs_frac may not reduce the usable season.
    out_season = np.where((season_mod < season_unobs_frac) | (season_mod > (1.0 - season_unobs_frac)))

    raw_obs = np.ones(unights.size)
    raw_obs[out_season] = 0

    cumulative_desired = ddf_slopes(ddf_name, raw_obs, night_season)

    night_mask = unights * 0
    night_mask[potential_nights] = 1

    unight_sched = match_cumulative(cumulative_desired, mask=night_mask)
    cumulative_sched = np.cumsum(unight_sched)

    nights_to_use = unights[np.where(unight_sched == 1)]

    # For each night, find the best time in the night.
    # XXX--probably need to expand this part to resolve the times when
    # multiple things get scheduled
    mjds = []
    for night_check in nights_to_use:
        in_night = np.where((night == night_check) & (np.isfinite(ddf_grid["%s_m5_g" % ddf_name])))[0]
        m5s = ddf_grid["%s_m5_g" % ddf_name][in_night]
        # we could intorpolate this to get even better than 15 min
        # resolution on when to observe
        max_indx = np.where(m5s == m5s.max())[0].min()
        mjds.append(ddf_grid["mjd"][in_night[max_indx]])

    return mjds, night_mjd, cumulative_desired, cumulative_sched


def generate_ddf_scheduled_obs(
    data_file=None,
    flush_length=2,
    mjd_tol=15,
    expt=30.0,
    alt_min=25,
    alt_max=85,
    HA_min=21.0,
    HA_max=3.0,
    sun_alt_max=-18,
    moon_min_distance=25.0,
    dist_tol=3.0,
    season_unobs_frac=0.1,
    nvis_master=[8, 10, 20, 20, 24, 18],
    filters="ugrizy",
    nsnaps=[1, 2, 2, 2, 2, 2],
    mjd_start=None,
    survey_length=10.0,
    sequence_time=60.0,
):
    """

    Parameters
    ----------
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
    season_unobs_frac : `float` (0.1)
        What fraction of the season should the DDF be considered
        unobservable. Taken off both the  start and end of the year,
        so a season frac of 0.1 means 20% of the time the DDF is
        considered unobservable, so it will be in-season for 9.6 months.
    nvis_master : list of ints ([8, 10, 20, 20, 24, 18])
        The number of visits to make per filter
    filters : `str` (ugrizy)
        The filter names.
    nsnaps : `list of ints` ([1, 2, 2, 2, 2, 2])
        The number of snaps to use per filter
    mjd_start : `float`
        Starting MJD of the survey. Default None, which calls
        rubin_sim.utils.survey_start_mjd
    survey_length : `float`
        Length of survey (years). Default 10.
    """
    if data_file is None:
        data_file = os.path.join(get_data_dir(), "scheduler", "ddf_grid.npz")

    if mjd_start is None:
        mjd_start = survey_start_mjd()

    flush_length = flush_length  # days
    mjd_tol = mjd_tol / 60 / 24.0  # minutes to days
    expt = expt
    alt_min = np.radians(alt_min)
    alt_max = np.radians(alt_max)
    dist_tol = np.radians(dist_tol)
    sun_alt_max = np.radians(sun_alt_max)
    moon_min_distance = np.radians(moon_min_distance)

    ddfs = ddf_locations()
    ddf_data = np.load(data_file)
    ddf_grid = ddf_data["ddf_grid"].copy()

    mjd_max = mjd_start + survey_length * 365.25

    # check if our pre-computed grid is over the time range we think
    # we are scheduling for
    if (ddf_grid["mjd"].min() > mjd_start) | (ddf_grid["mjd"].max() < mjd_max):
        warnings.warn("Pre-computed DDF properties don't match requested survey times")

    in_range = np.where((ddf_grid["mjd"] >= mjd_start) & (ddf_grid["mjd"] <= mjd_max))
    ddf_grid = ddf_grid[in_range]

    all_scheduled_obs = []
    for ddf_name in ["ELAISS1", "XMM_LSS", "ECDFS", "COSMOS", "EDFS_a"]:
        print("Optimizing %s" % ddf_name)

        mjds = optimize_ddf_times(
            ddf_name,
            ddfs[ddf_name][0],
            ddf_grid,
            season_unobs_frac=season_unobs_frac,
            sequence_time=sequence_time,
        )[0]
        for mjd in mjds:
            for filtername, nvis, nexp in zip(filters, nvis_master, nsnaps):
                if "EDFS" in ddf_name:
                    obs = scheduled_observation(n=int(nvis / 2))
                    obs["RA"] = np.radians(ddfs[ddf_name][0])
                    obs["dec"] = np.radians(ddfs[ddf_name][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt
                    obs["filter"] = filtername
                    obs["nexp"] = nexp
                    obs["note"] = "DD:%s" % ddf_name
                    obs["target"] = ddf_name

                    obs["mjd_tol"] = mjd_tol
                    obs["dist_tol"] = dist_tol
                    # Need to set something for HA limits
                    obs["HA_min"] = HA_min
                    obs["HA_max"] = HA_max
                    obs["alt_min"] = alt_min
                    obs["alt_max"] = alt_max
                    obs["sun_alt_max"] = sun_alt_max
                    all_scheduled_obs.append(obs)

                    obs = scheduled_observation(n=int(nvis / 2))
                    obs["RA"] = np.radians(ddfs[ddf_name.replace("_a", "_b")][0])
                    obs["dec"] = np.radians(ddfs[ddf_name.replace("_a", "_b")][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt
                    obs["filter"] = filtername
                    obs["nexp"] = nexp
                    obs["note"] = "DD:%s" % ddf_name.replace("_a", "_b")
                    obs["target"] = ddf_name.replace("_a", "_b")

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
                    obs = scheduled_observation(n=nvis)
                    obs["RA"] = np.radians(ddfs[ddf_name][0])
                    obs["dec"] = np.radians(ddfs[ddf_name][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt
                    obs["filter"] = filtername
                    obs["nexp"] = nexp
                    obs["note"] = "DD:%s" % ddf_name
                    obs["target"] = ddf_name

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
    # Put in the scripted ID so it's easier to track which ones fail.
    result["scripted_id"] = np.arange(result.size)
    return result
