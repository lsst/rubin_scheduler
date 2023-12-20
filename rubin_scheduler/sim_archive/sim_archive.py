import datetime
import hashlib
import json
import os
import shutil
import socket
import sys
from contextlib import redirect_stdout
from numbers import Number
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from astropy.time import Time
from conda.cli.main_list import print_packages
from conda.gateways.disk.test import is_conda_environment
from lsst.resources import ResourcePath

import rubin_scheduler
from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.utils import Site

SITE = None


def make_sim_archive_dir(
    observations,
    reward_df=None,
    obs_rewards=None,
    in_files={},
    sim_runner_kwargs={},
    tags=[],
    label=None,
    data_path=None,
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

    # Metadata
    # To use a different site, a user can set the global variable SITE.
    site = Site(name="LSST") if SITE is None else SITE

    def evening_local_date(mjd, longitude=site.longitude):
        evening_local_mjd = np.floor(mjd + longitude / 360 - 0.5).astype(int)
        evening_local_iso = Time(evening_local_mjd, format="mjd").iso[:10]
        return evening_local_iso

    opsim_metadata = {}
    opsim_metadata["scheduler_version"] = rubin_scheduler.__version__
    opsim_metadata["host"] = socket.getfqdn()
    opsim_metadata["username"] = os.environ["USER"]

    simulation_dates = {}
    if "mjd_start" in sim_runner_kwargs:
        simulation_dates["start"] = evening_local_date(sim_runner_kwargs["mjd_start"])

        if "survey_length" in sim_runner_kwargs:
            simulation_dates["end"] = evening_local_date(
                sim_runner_kwargs["mjd_start"] + sim_runner_kwargs["survey_length"]
            )
    else:
        simulation_dates["start"] = evening_local_date(observations["mjd"].min())
        simulation_dates["end"] = evening_local_date(observations["mjd"].max())

    if len(sim_runner_kwargs) > 0:
        opsim_metadata["sim_runner_kwargs"] = {}
        for key, value in sim_runner_kwargs.items():
            match value:
                case bool() | Number() | str():
                    opsim_metadata["sim_runner_kwargs"][key] = value
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
                    found_ids.append(int(found_dir[:-1]))
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

        print(f"Copied {source_fname} to {destination_rpath}")

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
