.. py:currentmodule:: rubin_scheduler.scheduler

.. _fbs=arch:

======================
Scheduler Architecture
======================

The scheduler in `rubin_scheduler` contains many classes which work in concert
to provide flexible, configurable algorithms for choosing observations, taking
into account the current weather conditions and observatory capabilities.


Overview
^^^^^^^^

In both actual operations and in simulations, the `CoreScheduler <fbs-api.html#rubin_scheduler.scheduler.schedulers.CoreScheduler>`_ does the work
of accepting telemetry of weather conditions and observatory state
(`update_conditions()`), and determining the best choice for the next
observation (or observations) (`request_observation()`) by polling its lists of `Surveys <fbs-api.html#module-rubin_scheduler.scheduler.surveys>`_ .
It also accepts updates of completed observations (`add_observation()`) and
when desired, can flush its internal queue of awaiting observations
(`flush_queue()`).
The `CoreScheduler` is the primary interaction point when running the scheduler.

The `CoreScheduler <fbs-api.html#rubin_scheduler.scheduler.schedulers.CoreScheduler>`_ itself however, does not determine what are useful or
desireable observations. That job belongs to the `Surveys <fbs-api.html#module-rubin_scheduler.scheduler.surveys>`_. While there could be only a
single `Survey` in the `CoreScheduler`, typically there are many, configured
for different goals or methods of acquiring observations. The `Surveys` are
held in the `CoreScheduler.survey_lists` and are organized into "tiers" --
each tier contains a list of `Surveys`.

Each `Survey` object can `check_feasibility()` for a quick check on whether
current conditions match its requirements and `calc_reward_function()` for
the full calculation of the desirability of an observation under the
current conditions, based on the survey's configuration.

Whenever `CoreScheduler.request_observation()` is called, the `CoreScheduler`
travels through each tier in `survey_lists`. Within each tier, the `Survey`
which is both feasible and has the greatest reward value will be chosen to
generate the next observation. If no `Survey` within a given tier is feasible,
the next tier will be queried for feasibility; this will continue
through the tiers until a `Survey` which is feasible is found. Thus
`Surveys` in the first tiers hold priority over `Surveys` in
later tiers.

Once a specific `Survey` is chosen to request an observation, the
`Survey.generate_observations()` method will be called. This provides
the specifics of the requested observation (or observations - many `Surveys`
request more than one observation at the same time, once chosen).

A step through the workflow of the CoreScheduler at a given time
to generate an observation request looks like:

.. mermaid::

    flowchart TD
        A([Start]) ==>B[Update Conditions]
        B ==> C[Request Observation]
        C ==> D([Consider Surveys in Tier])
        D --> M( )
        M --> E[Check Survey]
        M --> F[Check Survey]
        M --> G[Check Survey]
        E --> N( )
        F --> N
        G --> N
        N --> H{Is Any Survey Feasible?}
        H == No? Next tier==> D
        H == Yes? ==> I([Select Survey in Tier with highest reward])
        I ==> J[Winning Survey Generates Observation]
        J ==> K([End])


After an observation is acquired and `CoreScheduler.add_observation()` is
called, the observation is also passed to each individual `Survey`; some
`Surveys` will record this observation and add it into their `Features`, while
others may ignore it, depending on their configurations.


Surveys
^^^^^^^

The customization of survey strategy with `rubin_scheduler` happens in
the `Survey` objects and in how they are placed into the tiers in the
`CoreScheduler.survey_lists`, and there is a wide variety in `Survey` object
options.
A `Survey` could be configured to observe pairs of visits at any point
over the sky (such as the `BlobSurvey`) or it could be designed to simply follow
a scripted list of RA/Dec/Filter visits at a given time (such as a `ScriptedSurvey`).
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

Each `Survey` can `check_feasibility()`, which provides a quick check on
whether the current conditions meet the `Survey` requirements as well as
`calc_reward_function()`, which calculates the desirability of an
observation under the current conditions.
The calculation of the feasibility or reward for a given survey is governed by its
`BasisFunctions <fbs-api.html#module-rubin_scheduler.scheduler.basis_functions>`_
and `Features <fbs-api.html#module-rubin_scheduler.scheduler.features>`_.
The `BasisFunctions` calculate values to contribute to the reward that
consider some aspect of the current conditions: a simple example is
the `SlewtimeBasisFunction` which calculates a `value` based on the slewtime
from the current telescope position to the desired location on the sky.
The `Features` track relevant information for that `Survey`,
such as how many observations have already been obtained or when the last
observation at a given pointing was acquired, and can be used by the
`BasisFunctions` for that `Survey`.

Most `Survey` contain a list of `BasisFunctions`, which are combined to determine
the overall reward for that `Survey`. Each `BasisFunction` for a `Survey` has
an associated weight; the total reward for the `Survey` is simply the weighted
sum of the `BasisFunction` values.

A few `Surveys` do not use `BasisFunctions`. A `ScriptedSurvey`, for example,
might just have a list of desired observations and
time windows for those observations. The reward in that case might simply be
a constant value if there are any desired observations with time windows that
overlap the current time.

A `Survey` is considered infeasible if `check_feasibility()` returns False.
It is also infeasible if the final reward value is `-Infinity`. The final
reward value of a `Survey` is typically an array, but can be a scalar in the
case of `Surveys` defined only at a single point (such as a `FieldSurvey`).

If chosen to generate an observation request, the `Survey` will return
either a single or a series of requested observations,
using `generate_observations()`. The specifics of these requested observations
include details added by its
`Detailers <fbs-api.html#module-rubin_scheduler.scheduler.detailers>`_)
which can add requested rotation angle or dithering positions, if
applicable.
