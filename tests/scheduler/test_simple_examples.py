import unittest

import numpy as np
from astropy.time import Time

from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import (
    get_ideal_model_observatory,
    simple_field_survey,
    simple_greedy_survey,
    simple_pairs_survey,
    update_model_observatory_sunset,
)
from rubin_scheduler.scheduler.schedulers import ComCamBandSched, CoreScheduler
from rubin_scheduler.utils import SURVEY_START_MJD


class TestSurveyConveniences(unittest.TestCase):
    def setUp(self) -> None:
        self.survey_start = np.floor(SURVEY_START_MJD) + 0.5
        self.day_obs_start = Time(self.survey_start, format="mjd", scale="utc").iso[:10]

    def test_model_observatory_conveniences(self):
        """Test the model observatory convenience functions."""

        # Just check that we can acquire a model observatory
        # and it is set up for the date expected.
        observatory = get_ideal_model_observatory(dayobs=self.day_obs_start, survey_start=self.survey_start)
        conditions = observatory.return_conditions()
        assert conditions.mjd == observatory.mjd
        # The model observatory automatically advanced to -12 deg sunset
        assert (conditions.mjd - self.survey_start) < 1

        newday = self.survey_start + 4
        new_dayobs = Time(newday, format="mjd", scale="utc").iso[:10]
        newday = Time(f"{new_dayobs}T12:00:00", format="isot", scale="utc").mjd
        observatory = get_ideal_model_observatory(dayobs=new_dayobs, survey_start=self.survey_start)
        conditions = observatory.return_conditions()
        assert (conditions.mjd - newday) < 1

        # And update observatory to sunset, using a band scheduler
        # that only has 'g' available
        band_sched = ComCamBandSched(illum_bins=np.arange(0, 101, 100), loaded_band_groups=(("g",)))
        observatory = update_model_observatory_sunset(observatory, band_sched, twilight=-18)
        assert observatory.observatory.current_band == "g"
        assert observatory.conditions.sun_alt < np.radians(18)

    def test_simple_greedy_survey(self):
        # Just test that it still instantiates and provides observations.
        observatory = get_ideal_model_observatory(dayobs=self.day_obs_start, survey_start=self.survey_start)
        greedy = [simple_greedy_survey(bandname="r")]
        scheduler = CoreScheduler(greedy)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, band_scheduler=None, sim_duration=0.7
        )
        # Current survey_start_mjd should produce ~1000 visits
        # but changing this to a short night could reduce the total number
        self.assertTrue(len(observations) > 650)
        self.assertTrue(observations["mjd"].max() - observations["mjd"].min() > 0.35)
        self.assertTrue(np.unique(observations["scheduler_note"]).size == 1)
        self.assertTrue(np.unique(observations["scheduler_note"])[0] == "simple greedy r")
        # Check that we tracked things appropriately
        self.assertTrue(
            len(observations) == scheduler.survey_lists[0][0].extra_features["ObsRecorded"].feature
        )
        self.assertTrue(
            np.abs(
                observations[-1]["mjd"]
                - scheduler.survey_lists[0][0].extra_features["LastObs"].feature["mjd"]
            )
            < 15 / 60 / 60 / 24
        )

    def test_simple_pairs_survey(self):
        # Just test that it still instantiates and provides observations.
        observatory = get_ideal_model_observatory(dayobs=self.day_obs_start, survey_start=self.survey_start)
        pairs = [simple_pairs_survey(bandname="r", bandname2="i")]
        scheduler = CoreScheduler(pairs)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, band_scheduler=None, sim_duration=0.7
        )
        # Current survey_start_mjd should produce over ~950 visits
        # but changing this to a short night could reduce the total number
        self.assertTrue(len(observations) > 650)
        self.assertTrue(observations["mjd"].max() - observations["mjd"].min() > 0.35)
        self.assertTrue(np.unique(observations["scheduler_note"]).size == 2)
        # Check that we tracked things appropriately
        self.assertTrue(
            len(observations) == scheduler.survey_lists[0][0].extra_features["ObsRecorded"].feature
        )
        self.assertTrue(
            np.abs(
                observations[-1]["mjd"]
                - scheduler.survey_lists[0][0].extra_features["LastObs"].feature["mjd"]
            )
            < 15 / 60 / 60 / 24
        )

    def test_simple_field_survey(self):
        # Just test that it still instantiates and provides observations.
        observatory = get_ideal_model_observatory(dayobs=self.day_obs_start, survey_start=self.survey_start)
        # Find a good field position
        conditions = observatory.return_conditions()
        ra = conditions.lmst * 180 / 12.0
        dec = -89.0
        field_name = "almost_pole"
        field = [
            simple_field_survey(
                field_ra_deg=ra, field_dec_deg=dec, field_name=field_name, science_program="BLOCK-TEST"
            )
        ]
        # Add a greedy survey backup because of the visit gap requirement in
        # the default field survey
        greedy = [simple_greedy_survey(bandname="r")]
        scheduler = CoreScheduler([field, greedy])
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, band_scheduler=None, sim_duration=0.7
        )
        # Current survey_start_mjd should produce over ~950 visits
        # but changing this to a short night could reduce the total number
        self.assertTrue(len(observations) > 650)
        self.assertTrue(observations["mjd"].max() - observations["mjd"].min() > 0.35)
        # Check some information about the observation notes and names
        self.assertTrue(np.unique(observations["scheduler_note"]).size == 2)
        self.assertTrue("almost_pole" in observations["scheduler_note"])
        self.assertTrue("almost_pole" in observations["target_name"])
        # Check that the field survey got lots of visits
        field_obs = observations[np.where(observations["target_name"] == "almost_pole")]
        self.assertTrue(field_obs.size > 200)
        self.assertTrue(np.all(field_obs["science_program"] == "BLOCK-TEST"))
        self.assertTrue(field[0].extra_features["ObsRecorded_note"].feature == field_obs.size)
        self.assertTrue(
            np.abs(field_obs[-1]["mjd"] - field[0].extra_features["LastObs_note"].feature["mjd"])
            < 15 / 60 / 60 / 24
        )


if __name__ == "__main__":
    unittest.main()
