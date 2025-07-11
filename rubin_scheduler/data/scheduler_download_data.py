__all__ = ("data_dict", "scheduler_download_data", "download_rubin_data", "DEFAULT_DATA_URL")

import argparse
import os
import warnings
from shutil import rmtree, unpack_archive

import requests
from requests.exceptions import ConnectionError
from tqdm.auto import tqdm

from .data_sets import data_versions, get_data_dir

DEFAULT_DATA_URL = "https://s3df.slac.stanford.edu/data/rubin/sim-data/rubin_sim_data/"


def data_dict():
    """Creates a `dict` for all data buckets and the tar file they map to.
    To create tar files and follow any sym links, run:
    `tar -chvzf maf_may_2021.tgz maf`

    Returns
    -------
    result : `dict`
        Data bucket filenames dictionary with keys/values:
        "name" - Data bucket name (`str`).
        "version" - Versioned file name (`str`).
    """
    file_dict = {
        "scheduler": "scheduler_2025_06_27.tgz",
        "site_models": "site_models_2023_10_02.tgz",
        "skybrightness_pre": "skybrightness_pre_2024_11_19.tgz",
        "utils": "utils_2023_11_02.tgz",
    }
    return file_dict


def scheduler_download_data(file_dict=None):
    """Download data."""

    if file_dict is None:
        file_dict = data_dict()
    parser = argparse.ArgumentParser(description="Download data files for rubin_sim package")
    parser.add_argument(
        "--versions",
        dest="versions",
        default=False,
        action="store_true",
        help="Report expected versions, then quit",
    )
    parser.add_argument(
        "--update",
        dest="update",
        default=False,
        action="store_true",
        help="Update versions of data on disk to match current",
    )
    parser.add_argument(
        "-d",
        "--dirs",
        type=str,
        default=None,
        help="Comma-separated list of directories to download",
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        default=False,
        action="store_true",
        help="Force re-download of data directory(ies)",
    )
    parser.add_argument(
        "--url_base",
        type=str,
        default=DEFAULT_DATA_URL,
        help="Root URL of download location",
    )
    parser.add_argument(
        "--tdqm_disable",
        dest="tdqm_disable",
        default=False,
        action="store_true",
        help="Turn off tdqm progress bar",
    )
    args = parser.parse_args()

    download_rubin_data(
        data_dict(),
        dirs=args.dirs,
        print_versions_only=args.versions,
        update=args.update,
        force=args.force,
        url_base=args.url_base,
        tdqm_disable=args.tdqm_disable,
    )


def download_rubin_data(
    file_dict,
    dirs=None,
    print_versions_only=False,
    update=False,
    force=False,
    url_base=DEFAULT_DATA_URL,
    tdqm_disable=False,
):
    """Download external data blobs

    Parameters
    ----------
    file_dict : `dict`
        A dict with keys of directory names and values of remote filenames.
    dirs : `list` [`str`]
        List of directories to download. Default (None) assumes they are
        in file_dict
    versions : `bool`
        If True, print the versions currently on disk. Default False.
    update : `bool`
        If True, update versions on disk to match expected 'current'.
        Default False.
    force : `bool`
        If True, replace versions on disk with new download. Default False.
    url_base : `str`
        The URL to use, default to DEFAULT_DATA_URL
    tdqm_disable : `bool`
        If True, disable the tdqm progress bar. Default False.
    """
    # file_dict = dictionary of current versions
    if dirs is None:
        dirs = file_dict.keys()
    else:
        dirs = dirs.split(",")

    # Figure out where the rubin_sim_data is or is going
    data_dir = get_data_dir()
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Get dictionary of versions which are available on-disk
    versions = data_versions()
    if versions is None:
        versions = {}

    # ONLY check versions and return exit code
    if print_versions_only:
        print("Versions on disk currently // versions expected for this release:")
        mismatch_dict = {}
        match = True
        for k in file_dict:
            print(f"{k} : {versions.get(k, '')} // {file_dict[k]}")
            if versions.get(k, "") != file_dict[k]:
                match = False
                mismatch_dict[k] = False
        if match:
            print("Versions are in sync")
            return 0
        else:
            print("Versions do not match. ")
            print(f"{','.join([k for k in mismatch_dict])} are not matching.")
            return 1

    version_file = os.path.join(data_dir, "versions.txt")

    # See if base URL is alive
    url_base = url_base
    fail_message = f"Could not connect to {url_base}. Check site is up?"
    try:
        r = requests.get(url_base)
    except ConnectionError:
        print(fail_message)
        exit()
    if r.status_code != requests.codes.ok:
        print(fail_message)
        exit()

    # Now do downloading for "dirs"
    for key in dirs:
        filename = file_dict[key]
        path = os.path.join(data_dir, key)
        # Do some thinking to see if we should download new data for key
        download_this_dir = True
        if os.path.isdir(path):
            if force:
                # Remove and update, regardless
                rmtree(path)
                warnings.warn("Removed existing directory %s, downloading new copy" % path)
            elif not update:
                # Just see if it exists on-disk and keep it if it does
                warnings.warn("Directory %s already exists, skipping download" % path)
                download_this_dir = False
            else:
                # Update only if necessary
                if versions.get(key, "") == file_dict[key]:
                    download_this_dir = False
                else:
                    rmtree(path)
                    warnings.warn("Removed existing directory %s, downloading updated version" % path)
        if download_this_dir:
            # Download file
            url = url_base + filename
            print("Downloading file: %s" % url)
            # Stream and write in chunks (avoid large memory usage)
            r = requests.get(url, stream=True)
            file_size = int(r.headers.get("Content-Length", 0))
            if file_size < 245:
                warnings.warn(f"{url} file size unexpectedly small.")
            # Download this size chunk at a time; reasonable guess
            block_size = 512 * 512 * 10
            progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True, disable=tdqm_disable)
            print(f"Writing to {os.path.join(data_dir, filename)}")
            with open(os.path.join(data_dir, filename), "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            # untar in place
            unpack_archive(os.path.join(data_dir, filename), data_dir)
            os.remove(os.path.join(data_dir, filename))
            versions[key] = file_dict[key]

    # Write out the new version info to the data directory
    with open(version_file, "w") as f:
        for key in versions:
            print(key + "," + versions[key], file=f)

    # Write a little table to stdout
    new_versions = data_versions()
    print("Current/updated data versions:")
    for k in new_versions:
        if len(k) <= 10:
            sep = "\t\t"
        else:
            sep = "\t"
        print(f"{k}{sep}{new_versions[k]}")
