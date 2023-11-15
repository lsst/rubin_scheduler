.. py:currentmodule:: rubin_scheduler.scheduler

.. _fbs:

=========
Scheduler
=========

The `rubin_scheduler.scheduler` module implements the scheduling algorithms for the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST), via the `feature based scheduler`.

Scripts to use the scheduler code to create a simulated survey can be
found in the github repo at
`lsst-sims/sims_featureScheduler_runs3.3
<https://github.com/lsst-sims/sims_featureScheduler_runs3.3>`_.
To simulate a full 10 years of observations, additional skybrightness
data files must be downloaded, as described in :ref:`Precalculated Skybrightness <skybrightness-pre>`, using

`rs_download_sky`

Running a typical simulation will take on the order of 6 hours to complete.

The scheduler outputs a sqlite database containing the pointing history of
the telescope, along with information about the conditions of each
observation (visit).
Description of the :ref:`schema for the output database <output-schema>`.

.. toctree::

    Simulation output schema <output-schema>