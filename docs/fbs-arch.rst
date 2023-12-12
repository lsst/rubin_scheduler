.. py:currentmodule:: rubin_scheduler.scheduler

.. _fbs=arch:

======================
Scheduler Overview
======================

The scheduler in ``rubin_scheduler`` contains many classes which work in concert
to provide flexible, configurable algorithms for choosing observations, taking
into account the current weather conditions and observatory capabilities.


CoreScheduler
^^^^^^^^^^^^^

In both actual operations and in simulations, the `CoreScheduler <fbs-api.html#rubin_scheduler.scheduler.schedulers.CoreScheduler>`_ does the work
of accepting telemetry of weather conditions and observatory state
(``update_conditions()``), and determining the best choice for the next
observation (or observations) (``request_observation()``) by polling its lists of `Surveys <fbs-api.html#module-rubin_scheduler.scheduler.surveys>`_ .
It also accepts updates of completed observations (``add_observation()``) and
when desired, can flush its internal queue of awaiting observations
(``flush_queue()``).
The ``CoreScheduler`` is the primary interaction point when running the scheduler.

The `CoreScheduler <fbs-api.html#rubin_scheduler.scheduler.schedulers.CoreScheduler>`_
itself however, does not determine what are useful or
desireable observations. That job belongs to the
`Surveys <fbs-api.html#module-rubin_scheduler.scheduler.surveys>`_.
While there could be only a
single ``Survey`` in the ``CoreScheduler``, typically there are many, each
configured for different goals or methods of acquiring observations.
The ``Surveys`` are held in the ``CoreScheduler.survey_lists`` and are
organized into "tiers" -- each tier contains a list of ``Surveys``.

Each ``Survey`` object can ``check_feasibility()`` for a quick check on whether
current conditions match its requirements and ``calc_reward_function()`` for
the full calculation of the desirability of an observation under the
current conditions, based on the survey's configuration.

Whenever ``CoreScheduler.request_observation()`` is called, the ``CoreScheduler``
travels through each tier in ``survey_lists``. Within each tier, the ``Survey``
which is both feasible and has the greatest reward value will be chosen to
generate the next observation. If no ``Survey`` within a given tier is feasible,
the next tier will be queried for feasibility; this will continue
through the tiers until a ``Survey`` which is feasible is found. Thus
``Surveys`` in the first tier have priority over ``Surveys`` in
later tiers.

Once a specific ``Survey`` is chosen to request an observation, the
``Survey.generate_observations()`` method will be called. This provides
the specifics of the requested observation or observations. Many ``Surveys``
will request a series of observations to be executed in sequence.

A step through the workflow of the CoreScheduler at a given time
to generate an observation request looks like:

.. mermaid::

    flowchart TD
        A([Start]) ==>B[[Update Conditions]]
        B ==> C[[Request Observation]]
        C ==> D([Consider Surveys in Tier])
        D --> E[Check Survey]
        D --> F[Check Survey]
        D --> G[Check Survey]
        E --> H{Is any Survey in Tier feasible?}
        F --> H
        G --> H
        H == Yes ==> J([Select Survey in Tier with highest reward])
        H == No ==> I([Go to Next tier]) ==> D
        J ==> K[Winning Survey Generates Observation]
        K ==> L([End])


After an observation is acquired by the observatory
and ``CoreScheduler.add_observation()`` is
called to register the visit, the observation is also passed to each
individual ``Survey``. Some ``Surveys`` will record this observation, while
others may ignore it, depending on their configurations.


Surveys
^^^^^^^

The customization of survey strategy with ``rubin_scheduler`` happens in
the ``Survey`` objects, as well as in how they are placed into the tiers in the
``CoreScheduler.survey_lists``. There is a wide variety in how different
``Survey`` objects behave or can be configured.
A ``Survey`` could be configured to observe pairs of visits at any point
over the sky (such as the `BlobSurvey <fbs-api.html#rubin_scheduler.scheduler.surveys.BlobSurvey>`_)
or it could be designed to simply
follow a scripted list of RA/Dec/Filter visits at pre-specified time windows
(such as a `ScriptedSurvey <fbs-api.html#rubin_scheduler.scheduler.surveys.ScriptedSurvey>`_).

Each ``Survey`` can ``check_feasibility()``, which provides a quick check on
whether the current conditions meet the ``Survey`` requirements as well as
``calc_reward_function()``, which calculates the desirability of an
observation under the current conditions.
The calculation of the feasibility or reward for a given survey is governed
entirely by how these functions are implemented within the specific
``Survey`` class.

A ``Survey`` is considered infeasible if ``check_feasibility()`` returns False.
It is also infeasible if the maximum final reward value is ``-Infinity``.
The final reward value of a ``Survey`` is typically an array,
but can be a scalar in the case of ``Surveys`` defined only at a single point
(such as a ``FieldSurvey``).

If chosen to generate an observation request, the ``Survey`` will return
either a single requested observation or a series of requested observations,
using ``generate_observations()``. The specific choice of these observations
(such as whether this is a single visit or a series of visits)
is determined by the specific ``Survey``'s implementation of this method.
The specifics of these requested observations include details added by its
`Detailers <fbs-api.html#module-rubin_scheduler.scheduler.detailers>`_
which add requested rotation angle or dithering positions, if
applicable.


Calculating Feasibility and Rewards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calculation of feasibility or reward is entirely dependant on how
these methods are implemented in the ``Survey`` class, and this can vary
depending on the intended use of the ``Survey``.

A `ScriptedSurvey <fbs-api.html#rubin_scheduler.scheduler.surveys.ScriptedSurvey>`_
or one derived from this class may just have a
single constant reward value, but determine feasibility depending on whether
any of its list of desired observations (which contain possible
time windows) overlap the current time.

Most ``Surveys`` do contain a list of
`BasisFunctions <fbs-api.html#module-rubin_scheduler.scheduler.basis_functions>`_
which combine to calculate the overall reward for that ``Survey`` under
the current conditions. If so, each ``BasisFunction`` will also have
an associated weight; the total reward for the ``Survey`` is
simply the weighted sum of the ``BasisFunction`` values. These ``Surveys`` will
generally also have their own
`Features <fbs-api.html#module-rubin_scheduler.scheduler.features>`_, which
track relevant information for that ``Survey`` over time.

The ``BasisFunctions`` calculate values to contribute to the reward that
consider some aspect of the current conditions: a simple example is
the ``SlewtimeBasisFunction`` which calculates its ``value`` based on the slewtime
from the current telescope position to the desired location on the sky.
The ``Features`` track relevant information for that ``Survey``,
such as how many observations have already been obtained or when the last
observation at a given pointing was acquired, and can be used by the
``BasisFunctions`` for that ``Survey``.

Typically ``Survey`` classes which are intended to be used for large areas of
sky contain ``BasisFunctions`` and inherit from the
`BaseMarkovSurvey <fbs-api.html#rubin_scheduler.scheduler.surveys.BaseMarkovSurvey>`_.
Most of the observations in the current baseline come from a ``Survey`` class
in this category, the
`BlobSurvey <fbs-api.html#rubin_scheduler.scheduler.surveys.BlobSurvey>`_.


Basis Functions
---------------

For the ``Surveys`` which use ``BasisFunctions``, the ``BasisFunctions``
are where the list of "pros" and "cons" regarding obtaining observations under
the current conditions are calculated.
The final reward for these ``Surveys`` is the weighted sum of its
basis function values.

There are many different ``BasisFunctions`` available, and each can be configured
in different ways to generate different effects. Because they can be
configured in different ways, including keeping track of different
observations, ``BasisFunctions`` are not shared between ``Surveys``.
Some examples of common ``BasisFunctions`` include:

.. mermaid::

    classDiagram
        BasisFunction <|-- Slewtime
        BasisFunction <|-- M5Diff
        BasisFunction <|-- Footprint
        BasisFunction <|-- MoonAvoidance
        BasisFunction <|-- FilterLoaded
        BasisFunction : + Features
        BasisFunction : check_feasibility()
        BasisFunction : calc_value()
        BasisFunction : add_observation()
        BasisFunction : label()
        class Slewtime{
          + Short Slews
        }
        class M5Diff{
          + Better depth
        }
        class Footprint{
          + Uniform coverage
        }
        class MoonAvoidance{
          + Avoid the Moon
        }
        class FilterLoaded{
          + Filter Available
        }


The ``value`` of a given ``BasisFunction`` can be either a scalar or a map of the
sky (as `healpix <https://healpix.sourceforge.io/>`_ arrays). Generally, the
value returned depends on the type of ``BasisFunction``, although this can also
be modified by the properties of the ``Survey`` (``FieldSurveys``, for example,
only consider the ``BasisFunction`` value at the location of their target).

Most commonly, ``BasisFunctions`` return a map if they are considering a property
that varies across the sky, such as ``M5DiffBasisFunction`` which tracks current
skybrightness compared to the best possible skybrightness in the specified
filter. If the ``BasisFunction`` returns a value of ``-Infinity``, this will
be propagated through the weighted sum of ``BasisFunction`` values to the
``Survey`` reward value. This is easiest to understand with avoidance zone
masks like the ``MoonAvoidanceBasisFunction`` or the
``AvoidDirectWindBasisFunction`` which return ``-Infinity`` for the parts of the
sky which should be inaccessible for the telescope:
the ``-Infinity`` areas will be ``-Infinity`` in the ``Survey`` reward, and the
``Survey`` will not request observations in these parts of the sky.
If multiple ``BasisFunctions`` in a ``Survey`` have regions of ``-Infinity``,
it is possible for these regions to overlap in a way that makes the
final reward ``-Infinity`` at all points in the sky; this will make the
``Survey`` infeasible under those conditions.

Sometimes a ``BasisFunction`` returns a scalar value, such as for the
``FilterLoadedBasisFunction``. This ``BasisFunction`` tracks whether the filter
for a desired observation is available in the camera filter wheel. If the
filter is available, it returns ``0`` which doesn't modify the overall ``Survey``
reward. If the filter is not available, it returns ``-Infinity``, which
makes the ``Survey`` infeasible under those conditions.

