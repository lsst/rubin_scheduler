__all__ = ("get_data_dir", "data_versions")

import os


def get_data_dir():
    """Get the location of the rubin_sim data directory.

    Returns
    -------
    data_dir : `str`
        Path to the rubin_sim data directory.
    """
    # See if there is an environment variable with the path
    data_dir = os.getenv("RUBIN_SIM_DATA_DIR")

    # Set the root data directory
    if data_dir is None:
        data_dir = os.path.join(os.getenv("HOME"), "rubin_sim_data")
    return data_dir


def data_versions():
    """Get the dictionary of source filenames in the rubin_sim data directory.

    Returns
    -------
    result : `dict`
        Data directory filenames dictionary with keys/values:
        "name" - Data bucket name (`str`).
        "version" - Versioned file name (`str`).
    """
    data_dir = get_data_dir()
    result = None
    version_file = os.path.join(data_dir, "versions.txt")
    if os.path.isfile(version_file):
        with open(version_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        result = {}
        for line in content:
            ack = line.split(",")
            result[ack[0]] = ack[1]

    return result
