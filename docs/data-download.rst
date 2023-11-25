.. py:currentmodule:: rubin_scheduler.data

.. _data-download:

=============
Data Download
=============

The ``rubin_scheduler.data`` module provides scripts to download data
required to run the scheduler, as well as to check the expected
versions of the data.
It also provides a utility to interpret the location of
$RUBIN_SIM_DATA_DIR, checking both $HOME/rubin_sim_data and the
environment variable $RUBIN_SIM_DATA_DIR and returning the appropriate path.


Downloading Necessary Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the software in ``rubin_scheduler``, an additional approximately
500MB of data must be installed somewhere on the system. This directory
should be shared with ``rubin_sim`` and ``schedview`` if using those packages.

By default, the needed data files are downloaded to $HOME/rubin_sim_data.
If you would like the data to save elsewhere, you should set the
RUBIN_SIM_DATA_DIR environment variable.
In bash/zsh:

.. code-block:: bash

    export RUBIN_SIM_DATA_DIR="/my/preferred/data/path"

This environment variable, if used, should always be set before running
``rubin_scheduler`` packages. Another option is to add a sym-link to
your home directory:

.. code-block:: bash

    ln -s /my/preferred/data/path ~/rubin_sim_data


To download the appropriate data for your version of ``rubin_scheduler``:

.. code-block:: bash

    scheduler_download_data

This creates a series of directories at $RUBIN_SIM_DATA_DIR:

* scheduler (containing data used for setting up the scheduler footprint)
* site_models (containing data used for the weather histories)
* skybrightness_pre (containing a very minimal set of precalculated skybrightness data files)
* utils (containing data used for the utilities, such as an approximate LSST footprint map)

Note that the data will only be downloaded for the directories which do
not already exist, regardless of whether the version on disk is up to date.
To force an update to a version which matches the ``rubin_scheduler`` version:

.. code-block:: bash

    scheduler_download_data --update

This can also be applied only to certain directories, using the --dirs flag,
or to force an update of any or all directories using --force.

Downloading more extensive pre-calculated skybrightness files (such as
necessary to simulate a full survey) is covered in
:ref:`Downloading Skybrightness Data<skybrightness-pre>`.