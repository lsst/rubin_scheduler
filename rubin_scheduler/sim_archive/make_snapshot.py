__all__ = ["get_scheduler_instance_from_repo", "save_scheduler_cli"]

import argparse
import bz2
import gzip
import importlib.util
import lzma
import pickle
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

from git import Repo

from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler


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


def _save_scheduler(scheduler: CoreScheduler, file_name: str):
    if file_name.endswith(".bz2"):
        opener = bz2.open
    elif file_name.endswith(".xz"):
        opener = lzma.open
    elif file_name.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    with opener(file_name, "wb") as pio:
        pickle.dump(scheduler, pio)


def save_scheduler_cli(*args):
    parser = argparse.ArgumentParser(description="Create a scheduler pickle")
    parser.add_argument("pickle_fname", type=str, help="The file in which to save the scheduler.")
    parser.add_argument(
        "--repo", type=str, default=None, help="The repository from which to load the configuration."
    )
    parser.add_argument(
        "--script", type=str, default=None, help="The path to the config script (relative to the repo root)."
    )
    parser.add_argument(
        "--branch", type=str, default="main", help="The branch of the repo from which to get the script"
    )
    args = parser.parse_args() if len(args) == 0 else parser.parse_args(args)

    file_name = args.pickle_fname

    if args.repo is not None:
        scheduler = get_scheduler_instance_from_repo(
            config_repo=args.repo, config_script=args.script, config_branch=args.branch
        )
    else:
        example_scheduler_result = example_scheduler()
        if isinstance(example_scheduler_result, CoreScheduler):
            scheduler: CoreScheduler = example_scheduler_result
        else:
            # It might return a observatory, scheduler, observations tuple
            # instead.
            scheduler: CoreScheduler = example_scheduler_result[1]

    _save_scheduler(scheduler, file_name)


if __name__ == "__main__":
    save_scheduler_cli()
