__all__ = ("generate_ddf_scheduled_obs", "ddf_slopes", "match_cumulative", "optimize_ddf_times")

import os
import warnings

import numpy as np

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.utils import ScheduledObservationArray
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import SURVEY_START_MJD, calc_season, ddf_locations


def ddf_slopes(
    ddf_name,
    raw_obs,
    night_season,
    season_seq=30,
    min_season_length=0,
    boost_early_factor=None,
    boost_factor_third=2.0,
    boost_factor_fractional=0.0,
):
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
    offseason_length=73.05,
    low_season_frac=0,
    low_season_rate=0.3,
    mjd_start=SURVEY_START_MJD,
    season_seq=30,
    boost_early_factor=None,
    boost_factor_third=2,
    boost_factor_fractional=0.0,
):
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
    mjd_start : `float`, optional
        The MJD of the start of the survey. Used to identify the
        starting point when counting seasons.
        Default SURVEY_START_MJD.
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
    # expand sun mask backwards by the sequence time.
    n_back = np.ceil(sequence_time / (ddf_grid["mjd"][1] - ddf_grid["mjd"][0])).astype(int)
    shadow_indx = np.where(sun_mask == 0)[0] - n_back
    shadow_indx = shadow_indx[np.where(shadow_indx >= 0)]
    sun_mask[shadow_indx] = 0

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

    # Combine the masks.
    big_mask = sun_mask * airmass_mask * sky_mask * m5_mask

    # Identify which nights are useful to preschedule DDF visits.
    potential_nights = np.unique(night[np.where(big_mask > 0)])
    # prevent a repeat sequence in a night
    unights, indx = np.unique(night, return_index=True)
    night_mjd = ddf_grid["mjd"][indx]

    # Calculate season values for each night.
    night_season = calc_season(ddf_RA, night_mjd, mjd_start)
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
    raw_obs = np.ones(unights.size)
    raw_obs[out_season] = 0
    raw_obs[low_season] = low_season_rate

    cumulative_desired = ddf_slopes(
        ddf_name,
        raw_obs,
        night_season,
        season_seq=season_seq,
        boost_early_factor=boost_early_factor,
        boost_factor_third=boost_factor_third,
        boost_factor_fractional=boost_factor_fractional,
    )

    # Identify which nights (only scheduling 1 sequence per night)
    # would be usable, based on the masks above.
    night_mask = unights * 0
    night_mask[potential_nights] = 1
    # Calculate expected cumulative sequence schedule.
    unight_sched = match_cumulative(cumulative_desired, mask=night_mask)
    cumulative_sched = np.cumsum(unight_sched)

    nights_to_use = unights[np.where(unight_sched == 1)]

    # For each night, find the best time in the night and preschedule the DDF.
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
    nvis_master=[8, 10, 20, 20, 24, 18],
    bands="ugrizy",
    nsnaps=[1, 2, 2, 2, 2, 2],
    mjd_start=SURVEY_START_MJD,
    survey_length=10.0,
    sequence_time=60.0,
    offseason_length=36.525,
    low_season_frac=0,
    low_season_rate=0.3,
    ddf_kwargs=None,
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
    nvis_master : list of ints ([8, 10, 20, 20, 24, 18])
        The number of visits to make per band
    bands : `str` (ugrizy)
        The band names.
    nsnaps : `list of ints` ([1, 2, 2, 2, 2, 2])
        The number of snaps to use per band
    mjd_start : `float`
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
    ddf_kwargs : `dict`
        Dictionary to hold custom kwargs for each DDF. Default of None
        will use internal defaults and boost COSMOS early.
    """
    if data_file is None:
        data_file = os.path.join(get_data_dir(), "scheduler", "ddf_grid.npz")

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

    if ddf_kwargs is None:
        ddf_kwargs = {}
        ddf_kwargs["ELAISS1"] = {
            "season_seq": 30,
            "boost_early_factor": None,
            "boost_factor_third": 0,
            "offseason_length": offseason_length,
            "sequence_time": sequence_time,
            "low_season_frac": low_season_frac,
            "low_season_rate": low_season_rate,
        }

        ddf_kwargs["XMM_LSS"] = {
            "season_seq": 30,
            "boost_early_factor": None,
            "boost_factor_third": 0,
            "offseason_length": offseason_length,
            "sequence_time": sequence_time,
            "low_season_frac": low_season_frac,
            "low_season_rate": low_season_rate,
        }

        ddf_kwargs["ECDFS"] = {
            "season_seq": 30,
            "boost_early_factor": None,
            "boost_factor_third": 0,
            "offseason_length": offseason_length,
            "sequence_time": sequence_time,
            "low_season_frac": low_season_frac,
            "low_season_rate": low_season_rate,
        }

        ddf_kwargs["COSMOS"] = {
            "season_seq": 30,
            "boost_early_factor": 5.0,
            "boost_factor_third": 2,
            # Strange looking number for
            # mostly backwards compatibility
            "boost_factor_fractional": 2.42,
            "offseason_length": offseason_length,
            "sequence_time": sequence_time,
            "low_season_frac": low_season_frac,
            "low_season_rate": low_season_rate,
        }

        ddf_kwargs["EDFS_a"] = {
            "season_seq": 30,
            "boost_early_factor": None,
            "boost_factor_third": 0,
            "offseason_length": offseason_length,
            "sequence_time": sequence_time,
            "low_season_frac": low_season_frac,
            "low_season_rate": low_season_rate,
        }

    all_scheduled_obs = []
    for ddf_name in ddf_kwargs:
        print("Optimizing %s" % ddf_name)

        mjds = optimize_ddf_times(
            ddf_name,
            ddfs[ddf_name][0],
            ddf_grid,
            **ddf_kwargs[ddf_name],
        )[0]
        for mjd in mjds:
            for bandname, nvis, nexp in zip(bands, nvis_master, nsnaps):
                if "EDFS" in ddf_name:
                    obs = ScheduledObservationArray(n=int(nvis / 2))
                    obs["RA"] = np.radians(ddfs[ddf_name][0])
                    obs["dec"] = np.radians(ddfs[ddf_name][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt
                    obs["band"] = bandname
                    obs["nexp"] = nexp
                    obs["scheduler_note"] = "DD:%s" % ddf_name
                    obs["target_name"] = "DD:%s" % ddf_name

                    obs["mjd_tol"] = mjd_tol
                    obs["dist_tol"] = dist_tol
                    # Need to set something for HA limits
                    obs["HA_min"] = HA_min
                    obs["HA_max"] = HA_max
                    obs["alt_min"] = alt_min
                    obs["alt_max"] = alt_max
                    obs["sun_alt_max"] = sun_alt_max
                    all_scheduled_obs.append(obs)

                    obs = ScheduledObservationArray(n=int(nvis / 2))
                    obs["RA"] = np.radians(ddfs[ddf_name.replace("_a", "_b")][0])
                    obs["dec"] = np.radians(ddfs[ddf_name.replace("_a", "_b")][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt
                    obs["band"] = bandname
                    obs["nexp"] = nexp
                    obs["scheduler_note"] = "DD:%s" % ddf_name.replace("_a", "_b")
                    obs["target_name"] = "DD:%s" % ddf_name.replace("_a", "_b")
                    obs["science_program"] = "DD"
                    obs["observation_reason"] = "DD:%s" % ddf_name.replace("_a", "_b")

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
                    obs = ScheduledObservationArray(n=nvis)
                    obs["RA"] = np.radians(ddfs[ddf_name][0])
                    obs["dec"] = np.radians(ddfs[ddf_name][1])
                    obs["mjd"] = mjd
                    obs["flush_by_mjd"] = mjd + flush_length
                    obs["exptime"] = expt
                    obs["band"] = bandname
                    obs["nexp"] = nexp
                    obs["scheduler_note"] = "DD:%s" % ddf_name
                    obs["target_name"] = "DD:%s" % ddf_name
                    obs["science_program"] = "DD"
                    obs["observation_reason"] = "DD:%s" % ddf_name

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
