.. py:currentmodule:: rubin_scheduler.scheduler

.. _fbs:

=========
Scheduler
=========

The `rubin_scheduler.scheduler` module implements the scheduling algorithms for the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST), via the `feature based scheduler`.

Scripts to use the scheduler code to create a simulated survey can be
found in the github repo at
`lsst-sims/sims_featureScheduler_runs2.0
<https://github.com/lsst-sims/sims_featureScheduler_runs2.0>`_.
To be able to simulate a full 10 years of observations, additional skybrightness
data files must be downloaded (about 250GB), which can be done using the
script `rubin_scheduler/bin/rs_download_sky <https://github.com/lsst/rubin_scheduler/bin/rs_download_sky>`_.
Running a typical simulation will take on the order of 6 hours to complete.


The scheduler outputs a sqlite database containing the pointing history of
the telescope, along with information about the conditions of each
observation (visit).
Description of the :doc:`schema for the output database <output-schema>`.

.. toctree::

    Simulation output schema <output-schema>