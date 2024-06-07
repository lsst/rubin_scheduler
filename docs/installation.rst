.. py:currentmodule:: rubin_scheduler

.. _installation:

############
Installation
############

Quick Installation
------------------

Installation from PyPI:

::

    pip install rubin-scheduler

or from conda-forge:

::

    conda install -c conda-forge rubin-scheduler

Please note that following either installation,
additional data must be downloaded to use the software,
following the instructions at
:ref:`Data Download<data-download>`.

For Developer Use
-----------------

First, clone the `rubin_scheduler <https://github.com/lsst/rubin_scheduler>`_ repository:

::

 git clone git@github.com:lsst/rubin_scheduler.git
 cd rubin_scheduler
 conda create --channel conda-forge --name rubin_scheduler --file requirements.txt python=3.11
 conda activate rubin_scheduler
 conda install -c conda-forge --file=test-requirements.txt # Optional for running unit tests
 pip install -e . --no-deps

Please note that following installation,
additional data must be downloaded to use the software,
following the instructions at
:ref:`Data Download<data-download>`.


Building Documentation
----------------------

An online copy of the documentation is available at https://rubin-scheduler.lsst.io,
however building a local copy can be done as follows:

::

 pip install "documenteer[guide]"
 cd docs
 make html


The root of the local documentation will then be ``docs/_build/html/index.html``.

