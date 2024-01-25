import importlib
import logging

HAVE_LSST_RESOURCES = importlib.util.find_spec("lsst") and importlib.util.find_spec("lsst.resources")
if HAVE_LSST_RESOURCES:
    from .sim_archive import *
else:
    logging.error("rubin_scheduler.sim_archive requires lsst.resources.")
