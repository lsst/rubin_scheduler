"""Tools for maintaining an archive of opsim output and metadata.
"""

__all__ = [
    "make_sim_archive_dir",
    "transfer_archive_dir",
    "check_opsim_archive_resource",
    "read_archived_sim_metadata",
    "make_sim_archive_cli",
    "drive_sim",
]

import argparse
import datetime
import hashlib
import json
import logging
import lzma
import os
import pickle
import shutil
import socket
import sys
from contextlib import redirect_stdout
from numbers import Integral, Number
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time
from conda.cli.main_list import print_packages
from conda.gateways.disk.test import is_conda_environment

import rubin_scheduler
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.utils import SchemaConverter

LOGGER = logging.getLogger(__name__)

try:
    from lsst.resources import ResourcePath
except ModuleNotFoundError:
    LOGGER.error("Module lsst.resources required to use rubin_scheduler.sim_archive.")


def make_sim_archive_dir(
    observations,
    reward_df=None,
    obs_rewards=None,
    in_files={},
    sim_runner_kwargs={},
    tags=[],
    label=None,
    data_path=None,
    capture_env=True,
):
    """Create or fill a local simulation archive directory.

    Parameters
    ----------
    observations : `numpy.recarray`
        The observations data, in the "obs" format as accepted and created by
        `rubin_scheduler.scheduler.utils.SchemaConverter`.
    reward_df : `pandas.DataFrame`, optional
        The reward data, by default None.
    obs_rewards : `pandas.DataFrame`, optional
        The observation rewards data, by default None.
    in_files : `dict`, optional
        Additional input files to be included in the archive, by default {}.
    sim_runner_kwargs : `dict`, optional
        Additional simulation runner keyword arguments, by default {}.
    tags : `list` [`str`], optional
        A list of tags/keywords to be included in the metadata, by default [].
    label : `str`, optional
        A label to be included in the metadata, by default None.
    data_path : `str` or `pathlib.Path`, optional
        The path to the simulation archive directory, by default None.
    capture_env : `bool`
        Use the current environment as the sim environment.
        Defaults to True.

    Returns
    -------
    data_dir : `pathlib.Path` or `tempfile.TemporaryDirectory`
        The temporary directory containing the simulation archive.
    """
    if data_path is None:
        data_dir = TemporaryDirectory()
        data_path = Path(data_dir.name)
    else:
        data_dir = None

        if not isinstance(data_path, Path):
            data_path = Path(data_path)

    files = {}

    # Save the observations
    files["observations"] = {"name": "opsim.db"}
    opsim_output_fname = data_path.joinpath(files["observations"]["name"])
    SchemaConverter().obs2opsim(observations, filename=opsim_output_fname)

    # Save the rewards
    if reward_df is not None and obs_rewards is not None:
        files["rewards"] = {"name": "rewards.h5"}
        rewards_fname = data_path.joinpath(files["rewards"]["name"])
        if reward_df is not None:
            reward_df.to_hdf(rewards_fname, "reward_df")
        if obs_rewards is not None:
            obs_rewards.to_hdf(rewards_fname, "obs_rewards")

    # Save basic statistics
    files["statistics"] = {"name": "obs_stats.txt"}
    stats_fname = data_path.joinpath(files["statistics"]["name"])
    with open(stats_fname, "w") as stats_io:
        print(SchemaConverter().obs2opsim(observations).describe().T.to_csv(sep="\t"), file=stats_io)

    if capture_env:
        # Save the conda environment
        conda_prefix = Path(sys.executable).parent.parent.as_posix()
        if is_conda_environment(conda_prefix):
            conda_base_fname = "environment.txt"
            environment_fname = data_path.joinpath(conda_base_fname).as_posix()

            # Python equivilest of conda list --export -p $conda_prefix > $environment_fname
            with open(environment_fname, "w") as environment_io:
                with redirect_stdout(environment_io):
                    print_packages(conda_prefix, format="export")

            files["environment"] = {"name": conda_base_fname}

        # Save pypi packages
        pypi_base_fname = "pypi.json"
        pypi_fname = data_path.joinpath(pypi_base_fname).as_posix()

        pip_json_output = os.popen("pip list --format json")
        pip_list = json.loads(pip_json_output.read())

        with open(pypi_fname, "w") as pypi_io:
            print(json.dumps(pip_list, indent=4), file=pypi_io)

        files["pypi"] = {"name": pypi_base_fname}

    # Add supplied files
    for file_type, fname in in_files.items():
        files[file_type] = {"name": Path(fname).name}
        try:
            shutil.copyfile(fname, data_path.joinpath(files[file_type]["name"]))
        except shutil.SameFileError:
            pass

    # Add file hashes
    for file_type in files:
        fname = data_path.joinpath(files[file_type]["name"])
        with open(fname, "rb") as file_io:
            content = file_io.read()

        files[file_type]["md5"] = hashlib.md5(content).hexdigest()

    def convert_mjd_to_dayobs(mjd):
        # Use dayObs defn. from SITCOMTN-32: https://sitcomtn-032.lsst.io/
        evening_local_mjd = np.floor(mjd - 0.5).astype(int)
        evening_local_iso = Time(evening_local_mjd, format="mjd").iso[:10]
        return evening_local_iso

    opsim_metadata = {}
    if capture_env:
        opsim_metadata["scheduler_version"] = rubin_scheduler.__version__
        opsim_metadata["host"] = socket.getfqdn()

    opsim_metadata["username"] = os.environ["USER"]

    simulation_dates = {}
    if "mjd_start" in sim_runner_kwargs:
        simulation_dates["first"] = convert_mjd_to_dayobs(sim_runner_kwargs["mjd_start"])

        if "survey_length" in sim_runner_kwargs:
            simulation_dates["last"] = convert_mjd_to_dayobs(
                sim_runner_kwargs["mjd_start"] + sim_runner_kwargs["survey_length"] - 1
            )
    else:
        simulation_dates["first"] = convert_mjd_to_dayobs(observations["mjd"].min())
        simulation_dates["last"] = convert_mjd_to_dayobs(observations["mjd"].max())

    if len(sim_runner_kwargs) > 0:
        opsim_metadata["sim_runner_kwargs"] = {}
        for key, value in sim_runner_kwargs.items():
            # Cast numpy number types to ints, floats, and reals to avoid
            # confusing the yaml module.
            match value:
                case bool():
                    opsim_metadata["sim_runner_kwargs"][key] = value
                case Integral():
                    opsim_metadata["sim_runner_kwargs"][key] = int(value)
                case Number():
                    opsim_metadata["sim_runner_kwargs"][key] = float(value)
                case _:
                    opsim_metadata["sim_runner_kwargs"][key] = str(value)

    opsim_metadata["simulated_dates"] = simulation_dates
    opsim_metadata["files"] = files

    if len(tags) > 0:
        for tag in tags:
            assert isinstance(tag, str), "Tags must be strings."
        opsim_metadata["tags"] = tags

    if label is not None:
        assert isinstance(label, str), "The sim label must be a string."
        opsim_metadata["label"] = label

    sim_metadata_fname = data_path.joinpath("sim_metadata.yaml")
    with open(sim_metadata_fname, "w") as sim_metadata_io:
        print(yaml.dump(opsim_metadata, indent=4), file=sim_metadata_io)

    files["metadata"] = {"name": sim_metadata_fname}

    if data_dir is not None:
        # If we created a temporary directory, if we do not return it, it
        # will get automatically cleaned up, losing our work.
        # So, return it.
        return data_dir

    return data_path


def transfer_archive_dir(archive_dir, archive_base_uri="s3://rubin-scheduler-prenight/opsim/"):
    """Transfer the contents of an archive directory to an resource.

    Parameters:
    ----------
    archive_dir : `str`
        The path to the archive directory containing the files to be transferred.
    archive_base_uri : `str`, optional
        The base URI where the archive files will be transferred to. Default is "s3://rubin-scheduler-prenight/opsim/".

    Returns:
    -------
    resource_rpath : `ResourcePath`
        The destination resource.
    """

    metadata_fname = Path(archive_dir).joinpath("sim_metadata.yaml")
    with open(metadata_fname, "r") as metadata_io:
        sim_metadata = yaml.safe_load(metadata_io)

    insert_date = datetime.datetime.utcnow().date().isoformat()
    insert_date_rpath = ResourcePath(archive_base_uri).join(insert_date, forceDirectory=True)
    if not insert_date_rpath.exists():
        insert_date_rpath.mkdir()

    # Number the sims in the insert date dir by
    # looing for all the interger directories, and choosing the next one.
    found_ids = []
    for base_dir, found_dirs, found_files in insert_date_rpath.walk():
        if base_dir == insert_date_rpath:
            for found_dir in found_dirs:
                try:
                    found_dir_index = found_dir[:-1] if found_dir.endswith("/") else found_dir
                    found_ids.append(int(found_dir_index))
                except ValueError:
                    pass

    new_id = max(found_ids) + 1 if len(found_ids) > 0 else 1
    resource_rpath = insert_date_rpath.join(f"{new_id}", forceDirectory=True)
    resource_rpath.mkdir()

    # Include the metadata file itself.
    sim_metadata["files"]["metadata"] = {"name": "sim_metadata.yaml"}

    for file_info in sim_metadata["files"].values():
        source_fname = Path(archive_dir).joinpath(file_info["name"])
        with open(source_fname, "rb") as source_io:
            content = source_io.read()

        destination_rpath = resource_rpath.join(file_info["name"])
        destination_rpath.write(content)

        LOGGER.info(f"Copied {source_fname} to {destination_rpath}")

    return resource_rpath


def check_opsim_archive_resource(archive_uri):
    """Check the contents of an opsim archive resource.

    Parameters:
    ----------
    archive_uri : `str`
        The URI of the archive resource to be checked.

    Returns:
    -------
    validity: `dict`
        A dictionary of files checked, and their validity.
    """
    metadata_path = ResourcePath(archive_uri).join("sim_metadata.yaml")
    with metadata_path.open(mode="r") as metadata_io:
        sim_metadata = yaml.safe_load(metadata_io)

    results = {}

    for file_info in sim_metadata["files"].values():
        resource_path = ResourcePath(archive_uri).join(file_info["name"])
        content = resource_path.read()

        results[file_info["name"]] = file_info["md5"] == hashlib.md5(content).hexdigest()

    return results


def _build_archived_sim_label(base_uri, metadata_resource, metadata):
    label_base = metadata_resource.dirname().geturl().removeprefix(base_uri).rstrip("/").lstrip("/")

    # If a label is supplied by the metadata, use it
    if "label" in metadata:
        label = f"{label_base} {metadata['label']}"
        return label

    try:
        sim_dates = metadata["simulated_dates"]
        first_date = sim_dates["first"]
        last_date = sim_dates["last"]
        label = f"{label_base} of {first_date}"
        if last_date != first_date:
            label = f"{label} through {last_date}"
    except KeyError:
        label = label_base

    if "scheduler_version" in metadata:
        label = f"{label} with {metadata['scheduler_version']}"

    return label


def read_archived_sim_metadata(base_uri, latest=None, num_nights=5):
    """Read metadata for a time range of archived opsim output.

    Parameters:
    ----------
    base_uri : `str`
        The base URI of the archive resource to be checked.
    latest : `str`, optional
        The date of the latest simulation whose metadata should be loaded.
        This is the date on which the simulations was added to the archive,
        not necessarily the date on which the simulation was run, or any
        of the dates simulated.
        Default is today.
    num_nights : `int`
        The number of nights of the date window to load.

    Returns:
    -------
    sim_metadata: `dict`
        A dictionary of metadata for simulations in the date range.
    """
    latest_mjd = int(Time.now().mjd if latest is None else Time(latest).mjd)
    earliest_mjd = int(latest_mjd - num_nights)

    all_metadata = {}
    for mjd in range(earliest_mjd, latest_mjd + 1):
        iso_date = Time(mjd, format="mjd").iso[:10]
        date_resource = ResourcePath(base_uri).join(iso_date, forceDirectory=True)
        if date_resource.exists():
            for base_dir, found_dirs, found_files in date_resource.walk(file_filter=r".*sim_metadata.yaml"):
                for found_file in found_files:
                    found_resource = ResourcePath(base_dir).join(found_file)
                    these_metadata = yaml.safe_load(found_resource.read().decode("utf-8"))
                    these_metadata["label"] = _build_archived_sim_label(
                        base_uri, found_resource, these_metadata
                    )
                    all_metadata[str(found_resource.dirname())] = these_metadata

    return all_metadata


def make_sim_archive_cli(*args):
    parser = argparse.ArgumentParser(description="Add files to sim archive")
    parser.add_argument(
        "label",
        type=str,
        help="A label for the simulation.",
    )
    parser.add_argument(
        "opsim",
        type=str,
        help="File name of opsim database.",
    )
    parser.add_argument("--rewards", type=str, default=None, help="A rewards HDF5 file.")
    parser.add_argument(
        "--scheduler_version",
        type=str,
        default=None,
        help="The version of the scheduler that producte the opsim database.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="A snapshot of the scheduler used to produce the database, at the start of the simulation.",
    )
    parser.add_argument(
        "--script", type=str, default=None, help="The file name of the script run to create the simulation."
    )

    notebook_help = "The file name of the notebook run to create the simulation."
    notebook_help = notebook_help + " This can be produced using the %%notebook magic."
    parser.add_argument(
        "--notebook",
        type=str,
        default=None,
        help=notebook_help,
    )
    parser.add_argument(
        "--current_env",
        action="store_true",
        help="Record the current environment as the simulation environment.",
    )
    parser.add_argument(
        "--archive_base_uri",
        type=str,
        default="s3://rubin-scheduler-prenight/opsim/",
        help="Base URI for the archive",
    )
    parser.add_argument("--tags", type=str, default=[], nargs="*", help="The tags on the simulation.")
    arg_values = parser.parse_args() if len(args) == 0 else parser.parse_args(args)

    observations = SchemaConverter().opsim2obs(arg_values.opsim)

    if arg_values.rewards is not None:
        try:
            reward_df = pd.read_hdf(arg_values.rewards, "reward_df")
        except KeyError:
            reward_df = None

        try:
            obs_rewards = pd.read_hdf(arg_values.rewards, "obs_rewards")
        except KeyError:
            obs_rewards = None
    else:
        reward_df = None
        obs_rewards = None

    filename_args = ["scheduler", "script", "notebook"]
    in_files = {}
    for filename_arg in filename_args:
        try:
            filename = getattr(arg_values, filename_arg)
            if filename is not None:
                in_files[filename] = filename
        except AttributeError:
            pass

    data_path = make_sim_archive_dir(
        observations,
        reward_df,
        obs_rewards,
        in_files,
        tags=arg_values.tags,
        label=arg_values.label,
        capture_env=arg_values.current_env,
    )

    sim_archive_uri = transfer_archive_dir(data_path.name, arg_values.archive_base_uri)

    return sim_archive_uri


def drive_sim(
    observatory, scheduler, archive_uri=None, label=None, tags=[], script=None, notebook=None, **kwargs
):
    """Run a simulation and archive the results.

    Parameters
    ----------
    observatory : `ModelObservatory`
        The model for the observatory.
    scheduler : `CoreScheduler`
        The scheduler to use.
    archive_uri : `str`, optional
        The root URI of the archive resource into which the results should
        be stored. Defaults to None.
    label : `str`, optional
        The label for the simulation in the archive. Defaults to None.
    tags : `list` of `str`, optional
        The tags for the simulation in the archive. Defaults to an empty list.
    script : `str`
        The filename of the script producing this simulation.
        Defaults to None.
    notebook : `str`, optional
        The filename of the notebook producing the simulation.
        Defaults to None.

    Returns
    -------
    observatory : `ModelObservatory`
        The model for the observatory.
    scheduler : `CoreScheduler`
        The scheduler used.
    observations : `foo`
        The observations produced.
    reward_df : `pandas.DataFrame`, optional
        The table of rewards. Present if `record_rewards`
        or `scheduler.keep_rewards` is True.
    obs_rewards : `pandas.Series`, optional
        The mapping of entries in reward_df to observations. Present if
        `record_rewards` or `scheduler.keep_rewards` is True.
    resource_path : `ResourcePath`, optional
        The resource path to the archive of the simulation. Present if
        `archive_uri` was set.

    Notes
    -----
    Additional parameters not described above will be passed into
    `sim_runner`.

    If the `archive_uri` parameter is not supplied, `sim_runner` is run
    directly, so that `drive_sim` can act as a drop-in replacement of
    `sim-runner`.

    In a jupyter notebook, the notebook can be saved for the notebook paramater
    using `%notebook $notebook_fname` (where `notebook_fname` is variable
    holding the filename for the notebook) in the cell prior to calling
    `drive_sim`.
    """
    if "record_rewards" in kwargs:
        if kwargs["record_rewards"] and not scheduler.keep_rewards:
            raise ValueError("To keep rewards, scheduler.keep_rewards must be True")
    else:
        kwargs["record_rewards"] = scheduler.keep_rewards

    in_files = {}
    if script is not None:
        in_files["script"] = script

    if notebook is not None:
        in_files["notebook"] = notebook

    with TemporaryDirectory() as local_data_dir:
        # We want to store the state of the scheduler at the start of the sim,
        # so we need to save it now before we run the simulation.
        scheduler_path = Path(local_data_dir).joinpath("scheduler.pickle.xz")
        with lzma.open(scheduler_path, "wb", format=lzma.FORMAT_XZ) as pio:
            pickle.dump(scheduler, pio)
            in_files["scheduler"] = scheduler_path.as_posix()

        sim_results = sim_runner(observatory, scheduler, **kwargs)

        observations = sim_results[2]
        reward_df = sim_results[3] if scheduler.keep_rewards else None
        obs_rewards = sim_results[4] if scheduler.keep_rewards else None

        data_dir = make_sim_archive_dir(
            observations,
            reward_df=reward_df,
            obs_rewards=obs_rewards,
            in_files=in_files,
            sim_runner_kwargs=kwargs,
            tags=tags,
            label=label,
            capture_env=True,
        )

        if archive_uri is not None:
            resource_path = transfer_archive_dir(data_dir.name, archive_uri)
        else:
            resource_path = ResourcePath(data_dir.name, forceDirctory=True)

    results = sim_results + (resource_path,)
    return results
