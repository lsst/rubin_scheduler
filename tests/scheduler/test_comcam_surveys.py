import os
import unittest

import numpy as np
from astropy.time import Time

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import (
    get_comcam_sv_schedulers,
    get_model_observatory,
    get_sv_fields,
    update_model_observatory_sunset,
)
from rubin_scheduler.scheduler.schedulers import ComCamFilterSched

SAMPLE_BIG_DATA_FILE = os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")


class TestComCamSurveys(unittest.TestCase):

    @unittest.skipUnless(os.path.isfile(SAMPLE_BIG_DATA_FILE), "Test data not available.")
    def test_model_observatory_conveniences(self):
        """Test the model observatory convenience functions."""

        # Just check that we can acquire a model observatory
        # and it is set up for the date expected.
        dayobs = "2024-09-09"
        survey_start = Time(f"{dayobs}T12:00:00", format="isot", scale="utc").mjd
        observatory = get_model_observatory(dayobs=dayobs, survey_start=survey_start)
        conditions = observatory.return_conditions()
        assert conditions.mjd == observatory.mjd
        # The model observatory automatically advanced to -12 deg sunset
        assert (conditions.mjd - survey_start) < 1

        new_dayobs = "2024-09-12"
        newday = Time(f"{new_dayobs}T12:00:00", format="isot", scale="utc").mjd
        observatory = get_model_observatory(dayobs=new_dayobs, survey_start=survey_start)
        conditions = observatory.return_conditions()
        assert (conditions.mjd - newday) < 1

        # And update observatory to sunset, using a filter scheduler
        # that only has 'g' available
        filter_sched = ComCamFilterSched(illum_bins=np.arange(0, 101, 100), filter_sets=(("g",)))
        observatory = update_model_observatory_sunset(observatory, filter_sched, twilight=-18)
        assert observatory.observatory.current_filter == "g"
        assert observatory.conditions.sun_alt < np.radians(18)

    def test_comcam_sv_sched(self):
        """Test the comcam sv survey scheduler setup."""
        # This is likely to change as we go into commissioning,
        # so mostly I'm just going to test that the end result is
        # a usable scheduler
        dayobs = "2024-09-09"
        survey_start_mjd = Time(f"{dayobs}T12:00:00", format="isot", scale="utc").mjd
        scheduler, filter_scheduler = get_comcam_sv_schedulers()
        observatory = get_model_observatory(dayobs=dayobs, survey_start=survey_start_mjd)
        observatory = update_model_observatory_sunset(observatory, filter_scheduler)

        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, filter_scheduler, survey_length=30
        )
        assert len(observations) > 24000
        sv_fields = set(np.unique(observations["scheduler_note"]))
        all_sv_fields = set(list(get_sv_fields().keys()))
        # Probably won't observe all of the fields, but can do some.
        assert sv_fields.intersection(all_sv_fields) == sv_fields
