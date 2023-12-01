.. py:currentmodule:: rubin_scheduler.scheduler

.. _fbs=arch:

======================
Scheduler Architecture
======================

The scheduler in `rubin_scheduler` contains many classes which work in concert
to provide flexible, configurable algorithms for choosing observations, taking
into account the current weather conditions and observatory capabilities.

In both actual operations and in simulations, the `CoreScheduler <fbs-api.html#rubin_scheduler.scheduler.schedulers.CoreScheduler>`_ does the work
of accepting telemetry of weather conditions and observatory state
(`update_conditions()`) and then polling its lists of `Surveys <fbs-api.html#module-rubin_scheduler.scheduler.surveys>`_ to determine the best
choice for the next observation (`request_observation()`). It also accepts
updates of completed observations (`add_observation()`) and when desired,
can flush its internal queue of awaiting observations (`flush_queue()`).
The `CoreScheduler`, once configured, is the primary interaction point
for running the scheduler.

The `CoreScheduler <fbs-api.html#rubin_scheduler.scheduler.schedulers.CoreScheduler>`_ itself however, does not determine what are useful or
desireable observations. That job belongs to the `Surveys <fbs-api.html#module-rubin_scheduler.scheduler.surveys>`_. While there could be only a
single `Survey` in the `CoreScheduler`, typically there are many. These are
held in `CoreScheduler.survey_lists` and are organized into "tiers" --
each tier contains a list of `Surveys`.

A single Survey could be configured to observe pairs of visits at any point
over the sky (such as the `BlobSurvey`) or it could be designed to simply follow
a scripted list of RA/Dec/Filter visits at a given time (such as a `ScriptedSurvey`).
Each `Survey` object can `check_feasibility()` for a quick check on whether
current conditions match its requirements and `calc_reward_function()` for
the full calculation of the desirability of an observation from its configuration,
under the current conditions. The calculation of the feasibility or reward
for a given survey is governed by its
`BasisFunctions <fbs-api.html#module-rubin_scheduler.scheduler.basis_functions>`_
and `Features <fbs-api.html#module-rubin_scheduler.scheduler.features>`_.
The `BasisFunctions` contain the requirements and calculations for reward,
such accounting for slew time or the time since the last visit or the
expected limiting magnitude if an observation were to be acquired in the current
conditions, while the `Features` track relevant information for that `Survey`,
such as how many observations have already been obtained or when the last
observation at a given pointing was acquired.
Some `Survey` types include:

.. mermaid::

    classDiagram
        Survey <|-- FieldSurvey
        Survey <|-- ScriptedSurvey
        Survey <|-- GreedySurvey
        Survey <|-- BlobSurvey
        Survey : + BasisFunctions
        Survey : + Features
        Survey : + Detailers
        Survey : add_observations()
        Survey : check_feasibility()
        Survey : calc_reward_function()
        Survey : generate_observations()
        class FieldSurvey{
          + Target RA/Dec
        }
        class ScriptedSurvey{
          + [RA/Dec/Time/Filter]
          set_script()
        }
        class GreedySurvey{
          + Footprint
        }
        class BlobSurvey{
          + Footprint
          + "block" planning
          + Pairs
        }

Whenever `CoreScheduler.request_observation()` is called, the `CoreScheduler`
travels through each tier in `survey_lists`. Within each tier, the `Survey`
which is both feasible and has the greatest reward value will be chosen to
pick the next observation. If no `Survey` within a given tier is feasible,
the next tier will be queried for feasibility; this will continue
through the tiers until a `Survey` which is feasible is found.

Once a specific `Survey` is chosen to request an observation, the
`Survey.generate_observations()` method will be called. This provides
the specifics of the requested observation (or observations - many `Surveys`
request more than one observation at the same time, once chosen).
In filling out the specifics of a requested observation, the `Survey` will
call on its
`Detailers <fbs-api.html#module-rubin_scheduler.scheduler.detailers>`_)
to add information like requested rotation angle or dithering positions, if
applicable.

After an observation is acquired and `CoreScheduler.add_observation()` is
called, the observation is also passed to each individual `Survey`; some
`Surveys` will record this observation and add it into their `Features`, while
others may ignore it, depending on their configurations.



