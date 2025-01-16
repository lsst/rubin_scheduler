__all__ = ("sim_runner",)

import copy
import gzip
import pickle
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from rubin_scheduler.scheduler.schedulers import SimpleBandSched
from rubin_scheduler.scheduler.utils import ObservationArray, SchemaConverter, run_info_table
from rubin_scheduler.utils import Site, _approx_altaz2pa, pseudo_parallactic_angle, rotation_converter


def sim_runner(
    observatory,
    scheduler,
    band_scheduler=None,
    sim_start_mjd=None,
    sim_duration=3.0,
    filename=None,
    delete_past=True,
    n_visit_limit=None,
    step_none=15.0,
    verbose=True,
    extra_info=None,
    event_table=None,
    record_rewards=False,
    start_result_size=int(2e5),
    append_result_size=int(2.5e6),
    anomalous_overhead_func=None,
    telescope="rubin",
    snapshot_dir="",
):
    """
    run a simulation

    Parameters
    ----------
    survey_length : float (3.)
        The length of the survey ot run (days)
    step_none : float (15)
        The amount of time to advance if the scheduler fails to
        return a target (minutes).
    extra_info : dict (None)
        If present, dict gets added onto the information from the
        observatory model.
    event_table : np.array (None)
        Any ToO events that were included in the simulation
    record_rewards : bool (False)
        Save computed rewards
    start_result_size : int
        Size of observations array to pre-allocate at the start of
        the run. Default 2e5.
    append_result_size : int
        Size of observations array to append if start_result_size is
        too small. Default 2.5e6.
    anomalous_overhead_func: `Callable` or None
        A function or callable object that takes the visit time and slew time
        (in seconds) as argument, and returns and additional offset (also
        in seconds) to be applied as addinional overhead between exposures.
        Defaults to None.
    telescope : `str`
        Name of telecope for camera rotation. Default "rubin".
    snapshot_dir : `str`
        Dir in which to safe scheduler + pickle snapshots with
        each scheduler call. An empty string if the snapshots should not be
        saved. Defaults to an empty string.
    """

    if extra_info is None:
        extra_info = {}

    t0 = time.time()

    if band_scheduler is None:
        band_scheduler = SimpleBandSched()

    if sim_start_mjd is None:
        mjd = observatory.mjd + 0
        sim_start_mjd = mjd + 0
    else:
        mjd = sim_start_mjd + 0
        observatory.mjd = mjd

    sim_end_mjd = sim_start_mjd + sim_duration
    observations = ObservationArray(n=start_result_size)
    mjd_track = mjd + 0
    step = 1.0 / 24.0
    step_none = step_none / 60.0 / 24.0  # to days
    mjd_run = sim_end_mjd - sim_start_mjd
    nskip = 0

    mjd_last_flush = -1

    last_obs_queue_fill_mjd_ns = None
    obs_rewards = {}
    reward_dfs = []

    rc = rotation_converter(telescope=telescope)

    # Make sure correct bands are mounted
    conditions = observatory.return_conditions()
    bands_needed = band_scheduler(conditions)
    observatory.observatory.mount_bands(bands_needed)

    counter = 0
    while mjd < sim_end_mjd:
        # Save pickle here
        if snapshot_dir is not None and len(snapshot_dir) > 0:
            snapshot_dt_isot = Time(mjd, format="mjd").isot
            # The assert is needed to make type hint checkers happy, because
            # they cannot tell whether Time.isot will return a sting or an
            # array of strings. But, mjd here is a single value, so
            # Time.isot will be as well.
            if isinstance(snapshot_dt_isot, np.ndarray):
                assert len(snapshot_dt_isot) == 1
                snapshot_dt_isot = snapshot_dt_isot.item()

            assert isinstance(snapshot_dt_isot, str)
            snapshot_dt = snapshot_dt_isot.replace(":", "")
            snapshot_fname = Path(snapshot_dir).joinpath(f"sched_snapshot_{snapshot_dt}.p.gz").as_posix()
            # Save before updating the conditions to mimic observatory.
            # There is a significant tradeoff in compression: even a little
            # compression makes the save both much smaller and much slower.
            # A simple test on a USDF devel node, running one night:
            # open method                     time     size
            # open                            3m18s    173G
            # gzip.open(compresslevel=1)      16m56s   19G
            # lzma.open(preset=1)             48m33s   8.7G
            # lzma.open(preset=9)             4h20m31s 2.7G
            # For equivilent times, lzma always seems better, but its lowest
            # compression level is slower than gzip's lowest compression
            # level.
            # I also explored bzip2, but for any given compression level,
            # it was always slower then either gzip or lzma (or both).
            with gzip.open(snapshot_fname, "wb", compresslevel=1) as pio:
                pickle.dump([scheduler, observatory.return_conditions()], file=pio)

        if not scheduler._check_queue_mjd_only(observatory.mjd):
            scheduler.update_conditions(observatory.return_conditions())

        desired_obs = scheduler.request_observation(mjd=observatory.mjd)
        if record_rewards:
            if last_obs_queue_fill_mjd_ns != scheduler.queue_fill_mjd_ns:
                reward_dfs.append(scheduler.queue_reward_df)
                last_obs_queue_fill_mjd_ns = scheduler.queue_fill_mjd_ns

        if desired_obs is None:
            # No observation. Just step into the future and try again.
            warnings.warn("No observation. Step into the future and trying again.")
            observatory.mjd = observatory.mjd + step_none
            scheduler.update_conditions(observatory.return_conditions())
            nskip += 1
            # Check we didn't skip so far we should end
            if observatory.mjd > sim_end_mjd:
                break
            else:
                continue
        completed_obs, new_night = observatory.observe(desired_obs)

        if completed_obs is not None:
            if anomalous_overhead_func is not None:
                observatory.mjd += (
                    anomalous_overhead_func(completed_obs["visittime"], completed_obs["slewtime"]) / 86400
                )

            scheduler.add_observation(completed_obs)
            observations[counter] = completed_obs[0]
            band_scheduler.add_observation(completed_obs)
            counter += 1
            if counter == observations.size:
                add_observations = ObservationArray(n=append_result_size)
                observations = np.concatenate([observations, add_observations])

            if record_rewards:
                obs_rewards[completed_obs[0]["mjd"]] = last_obs_queue_fill_mjd_ns
        else:
            # An observation failed to execute, usually it was outside
            # the altitude limits.
            if observatory.mjd == mjd_last_flush:
                raise RuntimeError(
                    "Scheduler has failed to provide a valid observation multiple times "
                    f" at time ({observatory.mjd} from survey {scheduler.survey_index}."
                )
            # if this is a first offence, might just be that targets set.
            # Flush queue and try to get some new targets.
            scheduler.flush_queue()
            mjd_last_flush = copy.deepcopy(observatory.mjd)

        if new_night:
            # find out what bands we want mounted
            conditions = observatory.return_conditions()
            bands_needed = band_scheduler(conditions)
            observatory.observatory.mount_bands(bands_needed)

        mjd = observatory.mjd + 0
        if verbose:
            if (mjd - mjd_track) > step:
                progress = np.max((mjd - sim_start_mjd) / mjd_run * 100)
                text = "\rprogress = %.2f%%" % progress
                sys.stdout.write(text)
                sys.stdout.flush()
                mjd_track = mjd + 0
        if n_visit_limit is not None:
            if counter == n_visit_limit:
                break
        # XXX--handy place to interupt and debug
        # if len(observations) > 25:
        #    import pdb ; pdb.set_trace()

    # trim off any observations that were pre-allocated but not used
    observations = observations[0:counter]

    # Compute alt,az,pa, rottelpos for observations
    # Only warn if it's a low-accuracy astropy conversion
    lsst = Site("LSST")

    if len(observations) > 0:
        # Using pseudo_parallactic_angle, see https://smtn-019.lsst.io/v/DM-44258/index.html
        pa, alt, az = pseudo_parallactic_angle(
            np.degrees(observations["RA"]),
            np.degrees(observations["dec"]),
            observations["mjd"],
            lon=lsst.longitude,
            lat=lsst.latitude,
            height=lsst.height,
        )
        observations["alt"] = np.radians(alt)
        observations["az"] = np.radians(az)
        observations["pseudo_pa"] = np.radians(pa)
        observations["rotTelPos"] = rc._rotskypos2rottelpos(
            observations["rotSkyPos"], observations["pseudo_pa"]
        )

        # Also include traditional parallactic angle
        pa = _approx_altaz2pa(observations["alt"], observations["az"], lsst.latitude_rad)
        observations["pa"] = pa

    runtime = time.time() - t0
    print("Skipped %i observations" % nskip)
    print("Flushed %i observations from queue for being stale" % scheduler.flushed)
    print("Completed %i observations" % len(observations))
    print("ran in %i min = %.1f hours" % (runtime / 60.0, runtime / 3600.0))
    if len(observations) > 0:
        if filename is not None:
            print("Writing results to ", filename)
        if filename is not None:
            info = run_info_table(observatory, extra_info=extra_info)
            converter = SchemaConverter()
            converter.obs2opsim(observations, filename=filename, info=info, delete_past=delete_past)
        if event_table is not None:
            df = pd.DataFrame(event_table)
            con = sqlite3.connect(filename)
            df.to_sql("events", con)
            con.close()

        if record_rewards:
            reward_df = pd.concat(reward_dfs)
            obs_rewards_series = pd.Series(obs_rewards)
            obs_rewards_series.index.name = "mjd"
            obs_rewards_series.name = "queue_fill_mjd_ns"

            if filename is not None:
                with sqlite3.connect(filename) as con:
                    reward_df.to_sql("rewards", con)
                    obs_rewards_series.to_sql("obs_rewards", con)
    else:
        # Make sure there is something to return if there are no
        # observations
        reward_df = None
        obs_rewards_series = None

    if record_rewards:
        result = observatory, scheduler, observations, reward_df, obs_rewards_series
    else:
        result = observatory, scheduler, observations

    return result
