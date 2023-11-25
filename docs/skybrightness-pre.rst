.. py:currentmodule:: rubin_scheduler.skybrightness_pre

.. _skybrightness-pre:

=============================
Skybrightness (Precalculated)
=============================

The `rubin_scheduler.skybrightness_pre` module accesses pre-calculated
skybrightness values over the sky in each bandpass during the expected
on-sky period for LSST. The values calculated by the
``rubin_sim.skybrightness`` module are packaged into data files
which are then read and passed to
the scheduler by ``rubin_scheduler.skybrightness_pre``.


Downloading Skybrightness Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard rubin_sim_data download for skybrightness_pre contains a
small quantity of skybrightness data, limited to a period around the
survey start date. For full simulations, additional skybrightness
data will be needed. This can be downloaded to the
$RUBIN_SIM_DATA_DIR/skybrightness_pre directory using

.. code-block:: bash

    rs_download_sky

which will download a series of files from
`https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_skybrightness_pre/h5_2023_09_12/
<https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_skybrightness_pre/h5_2023_09_12/>`_.
These files are identified by the MJD range of data contained within each file (i.e. `60841_61054.h5` contains
pre-calculated skybrightness data covering MJD = 60841 to 61054).
Downloading individual files by hand, when only a limiting range of
skybrightness files are of interest, also works -- place the resulting
files in the $RUBIN_SIM_DATA_DIR/skybrightness_pre directory.

A full download of the skybrightness data files requires
approximately 75GB of disk space.
