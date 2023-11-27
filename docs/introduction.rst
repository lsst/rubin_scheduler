.. py:currentmodule:: rubin_scheduler

.. _introduction:

############
Introduction
############

The `Legacy Survey of Space and Time <http://www.lsst.org>`_ (LSST) is anticipated
to encompass around 2 million observations spanning a decade, averaging
800 visits per night. To streamline the acquisition of these observations,
we need a scheduler that is both fully automated and adaptable to weather conditions, observatory performance, and science priorities.
The "Feature Based Scheduler" (FBS) :ref:`scheduler <fbs>`  module in
`rubin_scheduler <https://github.com/lsst/rubin_scheduler>`_ encodes our
current best approach to achieving these objectives for
our all-sky multi-year survey.

These scheduling algorithms also need to be configured to best meet
the LSST science goals. Short and long-term survey
strategy choices must be evaluated using realistic simulations of the
expected pointing history. The :ref:`site_models <site-models>` and
:ref:`skybrightness_pre <skybrightness-pre>` modules support these simulations
by providing realistic weather telemetry, including seeing and cloud cover
histories and pre-calculated skybrightness maps. The simulations also include
a model of the observatory, providing accurate slew and settle times.
Configuration of the FBS for the baseline survey strategy can be found
in an ``example_scheduler`` in the ``scheduler`` module,
and configuration variations for experiments in survey strategy can
be found in various "sims_featureScheduler_runs*" github repositories
within  `lsst-sims`_.

.. _lsst-sims: https://github.com/lsst-sims/?q=sims_featureScheduler_runs&type=all&language=&sort=

Simulation of pointing histories for LSST can be carried out using
this software package alone. In operations, scheduling of the telescope
happens within a framework of other software packages which transmit
live telemetry of weather and observatory conditions as well as
carry out additional safety checks for each visit and coordinate with summit
databases to record relevant information, which are all necessary
for the actual scheduling of Rubin Observatory.
In both cases, the algorithms choosing the next visit are coming
from ``rubin_scheduler``.

When evaluating the outputs provided by ``rubin_scheduler``,
two additional software packages may be helpful:
`schedview <https://schedview.lsst.io>`_  which provides tools to
gain further insight into the scheduler itself and
`rubin_sim <https://rubin-sim.lsst.io>`_ which provides a framework
for writing and applying metrics to the output pointing databases.