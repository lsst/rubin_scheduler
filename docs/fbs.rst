.. py:currentmodule:: rubin_scheduler.scheduler

.. _fbs:

=========
Scheduler
=========

The ``rubin_scheduler.scheduler`` module implements the scheduling
algorithms for the Vera C. Rubin Observatory's
Legacy Survey of Space and Time (LSST),
via the ``feature based scheduler`` (FBS).

Tutorials for using the FBS are available in `Jupyter notebooks in the
rubin_sim_notebooks repository <https://github.com/lsst/rubin_sim_notebooks/tree/main/scheduler>`_.

Running a 10-year simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scripts to use the scheduler code to create a simulated survey can be
found in the github repo at
`lsst-sims/sims_featureScheduler_runs3.3
<https://github.com/lsst-sims/sims_featureScheduler_runs3.3>`_.
To simulate a full 10 years of observations, additional skybrightness
data files must be downloaded, as described in
:ref:`Downloading Skybrightness Data <skybrightness-pre>`, using

.. code-block:: bash

    rs_download_sky

Running a typical simulation will take on the order of 6 hours to complete.

Simulation Output Schema
^^^^^^^^^^^^^^^^^^^^^^^^

The scheduler outputs a sqlite database containing the pointing history of
the telescope, along with information about the conditions of each
observation (visit).
Description of the :ref:`schema for the output database <output-schema>`.

.. toctree::

    Simulation output schema <output-schema>