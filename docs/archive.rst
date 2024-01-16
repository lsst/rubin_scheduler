========================================================
The Prototype OpSim Archive for ``schedview`` Dashboards
========================================================

Introduction
------------

Several tools will require an archive that provides access to simulations provided by ``rubin_scheduler``.
For example, the prenight briefing dashboard supplied by ``schedview`` is a tool for visualizing simualtions of a night of observing, and it requires access to such simulations.
There will eventually be other dashboards that will also need to read such tables for visualization, as well as tables describing visits completed by the actual instruments.
Users of these dashboards will need to use them to select which data sets are to be visualised: the dashboard code needs to provide both user interface elements that let the user select the desired table of visits, and also actually load the table itself.
The dashboard servers run within containers on kubernetes-based infrastructure, which operate best when persistent data is stored outside the containers themselves.
Therefore, these dashboards require an external (to the container) resource that supports searching for available simulations, and access to the data itself.

This archive design is intended primarily as a prototype, something to experiment with for a better informed development of requiriments.
So, flexibility and speed of implementation have been prioritized, with the intention that there be a significant rofactoring (or even outright replacement) when requirements have been more thoroughly developed.

Design
------

The archive itself is a directory tree.
The URI of an archive is the URI of the root of the directory tree as supported by the ``lsst.resources`` package.
Each simulation in the archive has its own directory in this directory tree, named according to the ISO-8601 date on which a simulation was added to the archive, and a simple incrementing index separating different simulations added on the same day, such that the format for the directory for a specific simulation is::

    ${ARCHIVE_URI}/{ISO_DATE}/{ADDITION_INDEX}

For example, if the URI of the archive is::

    file:///my_data/sim_archive

then the URI of the third simulation added on June 21, 2030 will be::

    file:///my_data/sim_archive/2030-06-21/3

Each simulation directory contains a metadata yaml file named ``sim_metadata.yaml``.
In the above example, the URI for this metadata file would be::

    file:///my_data/sim_archive/2030-06-21/3/sim_metadata.yaml

A minimal ``sim_metadata.yaml`` file specifies the name of the sqlite3 database file with the visits.
For example, if the URI for the visit database in the above example is ``file:///my_data/sim_archive/2030-06-21/3/opsim.db``, then the minimal content of ``sim_metadata.yaml`` would be::

    files:
      observations:
        name: 'opsim.db'

All other data in the metadata file is optional, but additional metadata will be required if the archived simulation is to be used for some use cases.
For example, if ``schedview``'s ``prenight`` dashboard is to be able to load the reward data, it must be able to locate the reward data from the metadata file, so that the metadata file needs to look something like this::

    files:
      observations:
        name: 'opsim.db'
      rewards:
        name: 'rewards.h5'

Clients of the archive will also need to search available simulations for those meeting relevant criteria.
For example, the ``prenight`` dashboard will seach for simulations the include a desired night, in which case the range of nights covered by the simulation must be included.

A sample metadata file that includes an early guess at what the ``prenight`` dashboard will use looks like this::

    files:
        observations:
            name: opsim.db
        rewards:
            name: rewards.h5
    label: Notebook test on 2024-01-04 16:49:44.299
    simulated_dates:
        first: '2025-05-05'
        last: '2025-05-05'

In the above:

``label``
  Simulations will appear in drop-down section widgets in dashdoards such as the pre-night dashboard.
  The ``label`` element in the determines how the simulation will appear in the dropdown.
  In other applications, this element may also be used as plot annotations or column or row headings.

``simulation_dates``
  Shows the range of dates covered by the simulation.
  When the user specifies a night, the ``prenight`` dashboard will restrict the offered to those that cover the specified date.
  

Finally, a number of other elements may be included for debugging purposes.
A full file might look something like this::

    files:
        environment:
            md5: 4381d7cc82049141c70216121e39f56d
            name: environment.txt
        notebook:
            md5: 6b75c1dd8c4a3b83797c873f3270cc04
            name: notebook.ipynb
        observations:
            md5: 1909d1afaf744ee50bdcf1a9625826ab
            name: opsim.db
        pypi:
            md5: 9c86ea9b4e7aa40d3e206fad1a59ea31
            name: pypi.json
        rewards:
            md5: 6d3c9d3e0dd7764ed60312e459586e1b
            name: rewards.h5
        scheduler:
            md5: 5e88dfee657e6283dbc7a343f048db92
            name: scheduler.pickle.xz
        statistics:
            md5: c515ba27d83bdbfa9e65cdefff2d9d75
            name: obs_stats.txt
    label: Notebook test on 2024-01-04 16:49:44.299
    simulated_dates:
        first: '2025-05-05'
        last: '2025-05-05'
    scheduler_version: 1.0.1.dev25+gba1ca4d.d20240102
    sim_runner_kwargs:
        mjd_start: 60800.9565967191
        record_rewards: true
        survey_length: 0.5155218997970223
    tags:
    - notebook
    - devel
    host: neilsen-nb
    username: neilsen

This example has a number of additional elements useful for debugging, and which pehaps might be useful for future applictions, but which are not used (or planned to be used) by the prenight dashboard.

``files/*``
  A number of other types of files associated with specific simulations may be included.
  These may be useful in future applications, or for debugging only.
  See below for descriptions of the extra types of files in this example.
``files/${TYPE}/md5``
  Checksums for various files.
  These can be useful both for checking for corruption, and for determining whether two simulations are identical without needing to download either.
``scheduler_version``
  The version of the scheduler used to produce the simualtions.
``sim_runner_kwargs``
  The arguments to the execution of ``sim_runner`` used to run the simulation.
``tags``
  A list of ad-hoc keywords.
  For example, simulations used to test a specific jira issue may all have the name of the issue as a keyword.
  Simulations used to support a give tech note may have the name of the tech note.
``host``
  The hostname on which the simulation was run.
``username``
  The username of the user who ran the simulation.

Optional (for debugging or speculative future uses only) file types listed above are:

``environment``
  The conda environment specification for the environment used to run the simulation.
``notebook``
  The notebook used to create the simulation, for example as created using the ``%notebook`` jupyter magic.
``pypy``
  The ``pypy`` package list of the environment used to run the simulation.
  If the simulation is run using only conda-installed packages, this will be redundant with ``environment``.
``scheduler``
  A python pickle of the scheduler, in the state as of the start of the simulation.
``statistics``
  Basic statistics for the visit database.

