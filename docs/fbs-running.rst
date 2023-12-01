.. py:currentmodule:: rubin_scheduler.scheduler

.. _fbs-running

=========================
Running an FBS simulation
=========================

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

Shorter simulations can be created by setting the `--survey-length` to an
appropriate smaller value.

Additional tutorials that can help with understanding, configuring,
and running the scheduler are available  in `Jupyter notebooks in the
rubin_sim_notebooks repository
<https://github.com/lsst/rubin_sim_notebooks/tree/main/scheduler>`_.