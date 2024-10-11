__all__ = [
    "get_scheduler",
    "save_scheduler",
    "make_scheduler_snapshot",
    "make_scheduler_snapshot_cli",
    "get_scheduler_instance_from_repo",
    "snapshot_metadata",
]

import argparse
import bz2
import gzip
import importlib.util
import json
import lzma
import pickle
import sys
import types
import typing
import warnings
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from git import Repo

from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler


def _query_current_opsim_config_reference():
    current_week = date.today().isocalendar().week

    if current_week % 2 == 0:
        wkstr = f"Weeks {current_week-1}-{current_week}"
    else:
        wkstr = f"Weeks {current_week}-{current_week+1}"

    jql_str = f'Summary ~ "Support Summit Observing {wkstr}" ORDER BY createdDate DESC'

    from jira import JIRA

    jira = JIRA({"server": "https://rubinobs.atlassian.net/"})

    results = jira.search_issues(jql_str=jql_str, json_result=True, maxResults=2, fields="key")
    issue_keys = [i["key"] for i in results["issues"]]

    if len(issue_keys) < 1:
        warnings.warn("No issues found. Using head of develop instead.")
        git_reference = "develop"
    else:
        if len(issue_keys) > 1:
            warnings.warn("Multiple issues match: {', '.join(issue_keys)}, using the most recent")
        git_reference = f"tickets/{issue_keys[0]}"

    return git_reference


def get_scheduler_instance_from_repo(
    config_repo: str,
    config_script: str,
    config_branch: str = "main",
) -> CoreScheduler:
    """Generate a CoreScheduler according to a configuration in git.

    Parameters
    ----------
    config_repo : `str`
        The git repository with the configuration.
    config_script : `str`
        The configuration script path (relative to the repository root).
    config_branch : `str`, optional
        The branch of the repository to use, by default "main"

    Returns
    -------
    scheduler : `CoreScheduler`
        An instance of the Rubin Observatory FBS.

    Raises
    ------
    ValueError
        If the config file is invalid, or has invalid content.
    """

    with TemporaryDirectory() as local_config_repo_parent:
        repo: Repo = Repo.clone_from(config_repo, local_config_repo_parent, branch=config_branch)
        full_config_script_path: Path = Path(repo.working_dir).joinpath(config_script)
        config_module_name: str = "scheduler_config"
        config_module_spec = importlib.util.spec_from_file_location(
            config_module_name, full_config_script_path
        )
        if config_module_spec is None or config_module_spec.loader is None:
            # Make type checking happy
            raise ValueError(f"Cannot load config file {full_config_script_path}")

        config_module: types.ModuleType = importlib.util.module_from_spec(config_module_spec)
        sys.modules[config_module_name] = config_module
        config_module_spec.loader.exec_module(config_module)

    scheduler: CoreScheduler = config_module.get_scheduler()[1]
    return scheduler


def get_scheduler(
    config_repo: str | None,
    config_script: str | None,
    config_branch: str = "main",
) -> CoreScheduler:
    """Generate a CoreScheduler according to a configuration in git.

    Parameters
    ----------
    config_repo : `str`
        The git repository with the configuration.
    config_script : `str`
        The configuration script path (relative to the repository root).
    config_branch : `str`, optional
        The branch of the repository to use, by default "main". If set to
        ``jira``, try to determine the branch from a jira ticket.

    Returns
    -------
    scheduler : `CoreScheduler`
        An instance of the Rubin Observatory FBS.

    Raises
    ------
    ValueError
        If the config file is invalid, or has invalid content.
    """

    config_branch = _query_current_opsim_config_reference() if config_branch == "jira" else config_branch

    if config_repo is not None:
        if config_script is None:
            raise ValueError("If the config repo is set, the script must be as well.")
        scheduler = get_scheduler_instance_from_repo(
            config_repo=config_repo, config_script=config_script, config_branch=config_branch
        )
    else:
        example_scheduler_result = example_scheduler()
        if isinstance(example_scheduler_result, CoreScheduler):
            scheduler = example_scheduler_result
        else:
            # It might return a observatory, scheduler, observations tuple
            # instead.
            scheduler = example_scheduler_result[1]

    return scheduler


def save_scheduler(scheduler: CoreScheduler, file_name: str) -> None:
    """Save an instances of the scheduler in a pickle file,
    compressed according to its extension.

    Parameters
    ----------
    scheduler : `CoreScheduler`
        The scheduler to save.
    file_name : `str`
        The file in which to save the schedulers.
    """
    opener: typing.Callable = open

    if file_name.endswith(".bz2"):
        opener = bz2.open
    elif file_name.endswith(".xz"):
        opener = lzma.open
    elif file_name.endswith(".gz"):
        opener = gzip.open

    with opener(file_name, "wb") as pio:
        pickle.dump(scheduler, pio)

def snapshot_metadata(repo: str, script: str, git_reference: str) -> dict:
    """Generated a dictionary with keywords with scheduler config metadata.

    Parameters
    ----------
    repo : `str`
        The git repository with the configuration.
    script : `str`
        The configuration script path (relative to the repository root).
    git_reference : `str`, optional
        The branch of the repository to use.

    Returns
    -------
    metadata : `dict`
        The metadata dictionary
    """
    return {
        "opsim_config_repository": repo,
        "opsim_config_script": script,
        "opsim_config_branch": git_reference,
    }


def make_scheduler_snapshot(repo: str, script: str, git_reference: str) -> tuple[CoreScheduler, dict]:
    """Create a scheduler instance and metadata dictionary.

    Parameters
    ----------
    repo : `str`
        The git repository with the configuration.
    script : `str`
        The configuration script path (relative to the repository root).
    git_reference : `str`, optional
        The branch of the repository to use.

    Returns
    -------
    scheduler : `CoreScheduler`
        A scheduler instance.
    metadata : `dict`
        Metadata about the scheduler.
    """

    git_reference = _query_current_opsim_config_reference() if git_reference == "jira" else git_reference
    scheduler: CoreScheduler = get_scheduler(repo, script, git_reference)
    metadata: dict = snapshot_metadata(repo, script, git_reference)
    return scheduler, metadata


def make_scheduler_snapshot_cli(cli_args: list = []) -> None:
    parser = argparse.ArgumentParser(description="Create a scheduler pickle")
    parser.add_argument("--scheduler_fname", type=str, help="The file in which to save the scheduler.")
    parser.add_argument(
        "--repo", type=str, default=None, help="The repository from which to load the configuration."
    )
    parser.add_argument(
        "--script", type=str, default=None, help="The path to the config script (relative to the repo root)."
    )
    parser.add_argument(
        "--branch", type=str, default="main", help="The branch of the repo from which to get the script"
    )
    parser.add_argument("--metadata_fname", type=str, default="", help="file in which to save metadata.")
    args: argparse.Namespace = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    scheduler, metadata = make_scheduler_snapshot(args.repo, args.script, args.branch)
    save_scheduler(scheduler, args.scheduler_fname)
    if len(args.metadata_fname) > 0:
        with open(args.metadata_fname, "w") as metadata_io:
            print(json.dumps(metadata, indent=4), file=metadata_io)


if __name__ == "__main__":
    make_scheduler_snapshot_cli()
