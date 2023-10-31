.. py:currentmodule:: rubin_scheduler

.. _installation:

############
Installation
############

Quick Installation
------------------

Installation from PyPI:
``pip install rubin_scheduler``

or from conda-forge:
``conda install -c conda-forge rubin_scheduler``


For Developer Use
-----------------

First, clone the `rubin_scheduler` repository:

::

 git clone git@github.com:lsst/rubin_scheduler.git
 cd rubin_scheduler


Create a conda environment for it:

::

 conda create --channel conda-forge --name rubin_scheduler --file requirements.txt python=3.11


If you want to run tests, install the test requirements as well:

::

 conda activate rubin_scheduler
 conda install -c conda-forge --file=test-requirements.txt


Install the `rubin_scheduler` project into this environment (from the rubin_scheduler directory):

::

 pip install -e .




Building Documentation
----------------------

An online copy of the documentation is available at https://rubin-scheduler.lsst.io,
however building a local copy can be done as follows:

::

 pip install "documenteer[guide]"
 cd docs
 package-docs build


The root of the local documentation will then be `docs/_build/html/index.html`.

