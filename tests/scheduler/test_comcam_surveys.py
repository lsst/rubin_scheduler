import unittest

import numpy as np
from astropy.time import Time

from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import (
    get_comcam_sv_schedulers,
    get_model_observatory,
    get_sv_fields,
    update_model_observatory_sunset,
)
from rubin_scheduler.scheduler.schedulers import ComCamFilterSched
from rubin_scheduler.utils import survey_start_mjd


class TestComCamSurveys(unittest.TestCase):

    def test_model_observatory_conveniences(self):
        """Test the model observatory convenience functions."""

        # Just check that we can acquire a model observatory
        # and it is set up for the date expected.
        survey_start = survey_start_mjd()
        survey_start = np.floor(survey_start) + 0.5
        dayobs = Time(survey_start, format="mjd", scale="utc").iso[:10]
        observatory = get_model_observatory(dayobs=dayobs, survey_start=survey_start)
        conditions = observatory.return_conditions()
        assert conditions.mjd == observatory.mjd
        # The model observatory automatically advanced to -12 deg sunset
        assert (conditions.mjd - survey_start) < 1
        sun_ra_start = conditions.sun_ra_start
        mjd_start = observatory.mjd_start

        newday = survey_start + 4
        new_dayobs = Time(newday, format="mjd", scale="utc").iso[:10]
        newday = Time(f"{new_dayobs}T12:00:00", format="isot", scale="utc").mjd
        observatory = get_model_observatory(dayobs=new_dayobs, survey_start=survey_start)
        conditions = observatory.return_conditions()
        assert (conditions.mjd - newday) < 1
        # Check that advancing the day did not change the expected location
        # of the sun at the *start* of the survey
        assert conditions.mjd_start == mjd_start
        assert conditions.sun_ra_start == sun_ra_start

        # And update observatory to sunset, using a filter scheduler
        # that only has 'g' available
        filter_sched = ComCamFilterSched(illum_bins=np.arange(0, 101, 100), loaded_filter_groups=(("g",)))
        observatory = update_model_observatory_sunset(observatory, filter_sched, twilight=-18)
        assert observatory.observatory.current_filter == "g"
        assert observatory.conditions.sun_alt < np.radians(18)

    def test_comcam_sv_sched(self):
        """Test the comcam sv survey scheduler setup."""
        # This is likely to change as we go into commissioning,
        # so mostly I'm just going to test that the end result is
        # a usable scheduler
        survey_start = survey_start_mjd()
        survey_start = np.floor(survey_start) + 0.5
        dayobs = Time(survey_start, format="mjd", scale="utc").iso[:10]
        scheduler, filter_scheduler = get_comcam_sv_schedulers()
        observatory = get_model_observatory(dayobs=dayobs, survey_start=survey_start)
        observatory = update_model_observatory_sunset(observatory, filter_scheduler)

        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, filter_scheduler, survey_length=30
        )
        assert len(observations) > 24000
        sv_fields = set(np.unique(observations["scheduler_note"]))
        all_sv_fields = set(list(get_sv_fields().keys()))
        # Probably won't observe all of the fields, but can do some.
        assert sv_fields.intersection(all_sv_fields) == sv_fields


if __name__ == "__main__":
    unittest.main()
