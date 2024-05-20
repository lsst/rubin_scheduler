"""Tools for running the set of simulations used for a pre-night briefing."""

__all__ = [
    "AnomalousOverheadFunc",
    "run_prenights",
    "prenight_sim_cli",
]

import argparse
import io
import logging
import pickle
from datetime import datetime
from functools import partial
from tempfile import TemporaryFile
from typing import Callable, Optional, Sequence
from warnings import warn

import numpy as np
from astropy.time import Time
from matplotlib.pylab import Generator

from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler
from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.sim_archive.sim_archive import drive_sim
from rubin_scheduler.site_models import Almanac

try:
    from rubin_sim.data import get_baseline  # type: ignore
except ModuleNotFoundError:
    get_baseline = partial(warn, "Cannot find default baseline because rubin_sim is not installed.")

DEFAULT_ARCHIVE_URI = "s3://rubin-scheduler-prenight/opsim/"


def _run_sim(
    mjd_start: float,
    archive_uri: str,
    scheduler_io: io.BufferedRandom,
    label: str,
    tags: Sequence[str] = tuple(),
    sim_length: float = 2,
    anomalous_overhead_func: Optional[Callable] = None,
) -> None:
    logging.info(f"Running {label}.")

    observatory = ModelObservatory(
        cloud_data="ideal",
        seeing_data="ideal",
        downtimes="ideal",
    )

    scheduler_io.seek(0)
    scheduler = pickle.load(scheduler_io)

    drive_sim(
        observatory=observatory,
        scheduler=scheduler,
        archive_uri=archive_uri,
        label=label,
        tags=tags,
        script=__file__,
        mjd_start=mjd_start,
        survey_length=sim_length,
        record_rewards=True,
        anomalous_overhead_func=anomalous_overhead_func,
    )


def _mjd_now() -> float:
    # Used instead of just Time.now().mjd to make type checker happy.
    mjd = Time.now().mjd
    assert isinstance(mjd, float)
    return mjd


def _iso8601_now() -> str:
    # Used instead of just Time.now().mjd to make type checker happy.
    now_iso = Time.now().iso
    assert isinstance(now_iso, str)
    return now_iso


def _create_scheduler_io(
    day_obs_mjd: float,
    scheduler_fname: Optional[str] = None,
    scheduler_instance: Optional[CoreScheduler] = None,
    opsim_db=None,
) -> io.BufferedRandom:
    if scheduler_instance is not None:
        scheduler: CoreScheduler = scheduler_instance
    elif scheduler_fname is None:
        sample_scheduler = example_scheduler()
        if not isinstance(sample_scheduler, CoreScheduler):
            raise TypeError()

        scheduler: CoreScheduler = sample_scheduler
    else:
        with open(scheduler_fname, "rb") as sched_io:
            scheduler = pickle.load(sched_io)

    scheduler.keep_rewards = True

    if opsim_db is not None:
        last_preexisting_obs_mjd = day_obs_mjd + 0.5
        obs = SchemaConverter().opsim2obs(opsim_db)
        obs = obs[obs["mjd"] < last_preexisting_obs_mjd]
        if len(obs) > 0:
            scheduler.add_observations_array(obs)

    scheduler_io = TemporaryFile()
    pickle.dump(scheduler, scheduler_io)
    scheduler_io.seek(0)
    return scheduler_io


class AnomalousOverheadFunc:
    """Callable to return random overhead.

    Parameters
    ----------
    seed : `int`
        Random number seed.
    slew_scale : `float`
        The scale for the scatter in the slew offest (seconds).
    visit_scale : `float`, optional
        The scale for the scatter in the visit overhead offset (seconds).
        Defaults to 0.0.
    slew_loc : `float`, optional
        The location of the scatter in the slew offest (seconds).
        Defaults to 0.0.
    visit_loc : `float`, optional
        The location of the scatter in the visit offset (seconds).
        Defaults to 0.0.
    """

    def __init__(
        self,
        seed: int,
        slew_scale: float,
        visit_scale: float = 0.0,
        slew_loc: float = 0.0,
        visit_loc: float = 0.0,
    ) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.visit_loc: float = visit_loc
        self.visit_scale: float = visit_scale
        self.slew_loc: float = slew_loc
        self.slew_scale: float = slew_scale

    def __call__(self, visittime: float, slewtime: float) -> float:
        """Return a randomized offset for the visit overhead.

        Parameters
        ----------
        visittime : `float`
            The visit time (seconds).
        slewtime : `float`
            The slew time (seconds).

        Returns
        -------
        offset: `float`
            Random offset (seconds).
        """

        slew_overhead: float = slewtime * self.rng.normal(self.slew_loc, self.slew_scale)

        # Slew might be faster that expected, but it will never take negative
        # time.
        if (slewtime + slew_overhead) < 0:
            slew_overhead = 0.0

        visit_overhead: float = visittime * self.rng.normal(self.slew_loc, self.slew_scale)
        # There might be anomalous overhead that makes visits take longer,
        # but faster is unlikely.
        if visit_overhead < 0:
            visit_overhead = 0.0

        return slew_overhead + visit_overhead


def run_prenights(
    day_obs_mjd: float, archive_uri: str, scheduler_file: Optional[str] = None, opsim_db: Optional[str] = None
) -> None:
    """Run the set of scheduler simulations needed to prepare for a night.

    Parameters
    ----------
    day_obs_mjd : `float`
        The starting MJD>
    archive_uri : `str`
        The URI of visits completed before this night.
    scheduler_file : `str`, optional
        File from which to load the scheduler. None defaults to the example
        scheduler in rubin_sim, if it is installed.
        The default is None.
    opsim_db : `str`, optional
        The file name of the visit database for visits preceeding the
        simulation.
        The default is None.
    """

    exec_time: str = _iso8601_now()
    mjd_start: float = day_obs_mjd + 0.5
    scheduler_io = _create_scheduler_io(day_obs_mjd, scheduler_fname=scheduler_file, opsim_db=opsim_db)

    # Assign args common to all sims for this execution.
    run_sim = partial(_run_sim, archive_uri=archive_uri, scheduler_io=scheduler_io)

    # Begin with an ideal pure model sim
    run_sim(
        mjd_start,
        label=f"Nominal start and overhead, ideal conditions, run at {exec_time}",
        tags=["ideal", "nominal"],
    )

    # Delayed start
    # Almanac.get_sunset_info does not use day_obs, so just index
    # Almanac.sunsets for what we want directly.
    all_sun_n12_setting = Almanac().sunsets["sun_n12_setting"]
    before_day_obs = all_sun_n12_setting < day_obs_mjd + 0.5
    after_day_obs = all_sun_n12_setting > day_obs_mjd + 1.5
    on_day_obs = ~(before_day_obs | after_day_obs)
    sun_n12_setting = all_sun_n12_setting[on_day_obs].item()

    for minutes_delay in (0, 1, 10, 60, 240):
        delayed_mjd_start = sun_n12_setting + minutes_delay / (24.0 * 60)

        # 2 nights of observing, 0.5 nights to shift to the day_obs rollover
        sim_length = day_obs_mjd + 2.5 - delayed_mjd_start

        run_sim(
            delayed_mjd_start,
            label=f"Start time delayed by {minutes_delay} minutes,"
            + f" Nominal slew and visit overhead, ideal conditions, run at {exec_time}",
            tags=["ideal", f"delay_{minutes_delay}"],
            sim_length=sim_length,
        )

    # Run a few different scatters of visit time
    anomalous_overhead_scale = 0.1
    for anomalous_overhead_seed in range(101, 106):
        anomalous_overhead_func = AnomalousOverheadFunc(anomalous_overhead_seed, anomalous_overhead_scale)
        run_sim(
            mjd_start,
            label=f"Anomalous overhead {anomalous_overhead_seed, anomalous_overhead_scale},"
            + f" Nominal start, ideal conditions, run at {exec_time}",
            tags=["ideal", "anomalous_overhead"],
            anomalous_overhead_func=anomalous_overhead_func,
            sim_length=2,
        )


def _parse_dayobs_to_mjd(dayobs: str | float) -> float:
    try:
        day_obs_mjd = Time(dayobs).mjd
    except ValueError:
        try:
            day_obs_mjd = Time(datetime.strptime(str(dayobs), "%Y%m%d"))
        except ValueError:
            day_obs_mjd = Time(dayobs, format="mjd")

    if not isinstance(day_obs_mjd, float):
        raise ValueError

    return day_obs_mjd


def prenight_sim_cli(*args):
    if Time.now() >= Time("2025-05-01"):
        default_time = Time(int(_mjd_now() - 0.5), format="mjd")
    else:
        default_time = Time("2025-05-14")

    parser = argparse.ArgumentParser(description="Run prenight simulations")
    parser.add_argument(
        "--dayobs",
        type=str,
        default=default_time.iso,
        help="The day_obs of the night to simulate.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=DEFAULT_ARCHIVE_URI,
        help="Archive in which to store simulation results.",
    )
    parser.add_argument("--scheduler", type=str, default=None, help="pickle file of the scheduler to run.")

    # Only pass a default if we have an opsim
    opsim_arg_kwargs = {"type": str, "help": "Opsim database from which to load previous visits."}
    baseline = get_baseline()
    if baseline is not None:
        opsim_arg_kwargs["default"] = baseline

    parser.add_argument("--opsim", **opsim_arg_kwargs)
    args = parser.parse_args() if len(args) == 0 else parser.parse_args(args)

    day_obs_mjd = _parse_dayobs_to_mjd(args.dayobs)
    archive_uri = args.archive
    scheduler_file = args.scheduler
    opsim_db = args.opsim

    run_prenights(day_obs_mjd, archive_uri, scheduler_file, opsim_db)


if __name__ == "__main__":
    prenight_sim_cli()
