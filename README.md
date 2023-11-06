[![DOI](https://zenodo.org/badge/712606243.svg)](https://zenodo.org/doi/10.5281/zenodo.10076770)


# rubin_scheduler

Feature Based Scheduler for Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST).

This repository contains the scheduling algorithms for the LSST, as implemented in the Feature Based Scheduler (FBS). More documentation on the FBS is available at https://rubin-scheduler.lsst.io and in jupyter notebooks available in our [tutorials repository](https://github.com/lsst/rubin_sim_notebooks/tree/main/scheduler). 

# Install From Source

```
git clone https://github.com/lsst/rubin_scheduler.git ; cd rubin_scheduler  ## clone and cd into repo
conda create -n rubin-sim ; conda activate rubin-sim   ## optional (but recommended) new conda env
conda install -c conda-forge --file=requirements.txt  ## install dependencies
conda install -c conda-forge --file=test-requirements.txt  ## for running unit tests
pip install -e .
scheduler_download_data  ## Downloads ~500 MB of data to $RUBIN_SIM_DATA_DIR (~/rubin_sim_data if unset)
```

