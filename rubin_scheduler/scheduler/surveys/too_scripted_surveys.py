__all__ = ("ToOScriptedSurvey", "gen_too_surveys")

from copy import copy

import numpy as np

from rubin_scheduler.scheduler.surveys import BaseMarkovSurvey, ScriptedSurvey
from rubin_scheduler.scheduler.utils import (
    ScheduledObservationArray,
    comcam_tessellate,
    order_observations,
    thetaphi2xyz,
    xyz2thetaphi,
)
from rubin_scheduler.site_models import _read_fields
from rubin_scheduler.utils import _approx_ra_dec2_alt_az, _ra_dec2_hpid


def rotx(theta, x, y, z):
    """rotate the x,y,z points theta radians about x axis"""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xp = x
    yp = y * cos_t + z * sin_t
    zp = -y * sin_t + z * cos_t
    return xp, yp, zp


DEFAULT_EXP_TIME = 29.2


class ToOScriptedSurvey(ScriptedSurvey, BaseMarkovSurvey):
    """If there is a new ToO event, generate a set of scripted
    observations to try and follow it up.

    Parameters
    ----------
    times : list of floats
        The times after the detection that observations
        should be attempted (hours). Default [1, 2, 4, 24, 48]
    filters_at_times : list of str
        The filters that should be observed at each time in `times`.
        Default ["gz", "gz", "gz", "gz", "gz", "gz"]
    nvis : list of int
        The number of visits per filter at each time in `times`.
        Default [1, 1, 1, 1, 6, 6]
    exptimes : list of floats
        The exposure times to use for each time in `times`.
        Default [DEFAULT_EXP_TIME]*5,
    alt_min : float
        Do not attempt observations below this limit (degrees).
        Note the telescope alt limit is 20 degrees, however, slew
        and filter change time means alt_min here should be set higher
        (otherwise, target will pass altitude check, but then fail
        to observe by the time the telescope gets there).

    """

    def __init__(
        self,
        basis_functions,
        followup_footprint=None,
        nside=32,
        reward_val=1e6,
        times=[1, 2, 4, 24, 48],
        filters_at_times=["gz", "gz", "gz", "gz", "gz", "gz"],
        nvis=[1, 1, 1, 1, 6, 6],
        exptimes=[DEFAULT_EXP_TIME] * 6,
        camera="LSST",
        survey_name="ToO",
        flushtime=2.0,
        mjd_tol=1.0 / 24.0,
        dist_tol=0.5,
        alt_min=25.0,
        alt_max=85.0,
        HA_min=5,
        HA_max=19,
        ignore_obs=None,
        dither=True,
        seed=42,
        npositions=20000,
        n_snaps=2,
        n_usnaps=1,
        id_start=1,
        detailers=None,
        too_types_to_follow=[""],
        split_long=False,
    ):

        # Make sure lists all match
        check = np.unique([len(filters_at_times), len(times), len(nvis), len(exptimes)])
        if np.size(check) > 1:
            raise ValueError("lengths of times, filters_at_times, nvis, and exptimes must match.")

        # Figure out what else I need to super here

        if ignore_obs is None:
            ignore_obs = []

        self.basis_functions = basis_functions
        self.survey_name = survey_name
        self.followup_footprint = followup_footprint
        self.last_event_id = -1
        self.night = -1
        self.reward_val = reward_val
        self.times = np.array(times) / 24.0  # to days
        self.filters_at_times = filters_at_times
        self.exptimes = exptimes
        self.nvis = nvis
        self.n_snaps = n_snaps
        self.n_usnaps = n_usnaps
        self.nside = nside
        self.flushtime = flushtime / 24.0
        self.mjd_tol = mjd_tol
        self.dist_tol = np.radians(dist_tol)
        self.alt_min = np.radians(alt_min)
        self.alt_max = np.radians(alt_max)
        self.HA_min = HA_min
        self.HA_max = HA_max
        self.ignore_obs = ignore_obs
        self.extra_features = {}
        self.extra_basis_functions = {}
        self.detailers = []
        self.dither = dither
        self.id_start = id_start
        self.detailers = detailers
        self.last_mjd = -1
        self.too_types_to_follow = too_types_to_follow
        self.split_long = split_long

        self.camera = camera
        # Load the OpSim field tesselation and map healpix to fields
        if self.camera == "LSST":
            ra, dec = _read_fields()
            self.fields_init = np.empty(ra.size, dtype=list(zip(["RA", "dec"], [float, float])))
            self.fields_init["RA"] = ra
            self.fields_init["dec"] = dec
        elif self.camera == "comcam":
            self.fields_init = comcam_tessellate()
        else:
            ValueError('camera %s unknown, should be "LSST" or "comcam"' % camera)
        self.fields = self.fields_init.copy()

        self.hp2fields = np.array([])
        self._hp2fieldsetup(self.fields["RA"], self.fields["dec"])

        # Don't bother with checking if we can run before twilight ends
        self.before_twi_check = False

        # Initialize the list of scripted observations
        self.clear_script()

        # Generate and store rotation positions to use.
        # This way, if different survey objects are seeded the same, they will
        # use the same dither positions each night
        rng = np.random.default_rng(seed)
        self.lon = rng.random(npositions) * np.pi * 2
        # Make sure latitude points spread correctly
        # http://mathworld.wolfram.com/SpherePointPicking.html
        self.lat = np.arccos(2.0 * rng.random(npositions) - 1.0)
        self.lon2 = rng.random(npositions) * np.pi * 2
        self.spin_indx = 0

        # list to keep track of alerts we have already seen
        self.seen_alerts = []

    # Need to make sure new set_script call doesn't clobber old script!
    def set_script(self, obs_wanted, append=True):
        """
        Parameters
        ----------
        obs_wanted : rubin_scheduler.scheduler.utils.ScheduledObservationArray
            The observations that should be executed.
        append : bool
            Should the obs_wanted be appended to any script already set?
        """

        obs_wanted.sort(order=["mjd", "filter"])
        # Give each desired observation a unique "scripted ID". To be used for
        # matching and logging later.
        obs_wanted["scripted_id"] = np.arange(self.id_start, self.id_start + np.size(obs_wanted))
        # Update so if we set the script again the IDs will not be reused.
        self.id_start = np.max(obs_wanted["scripted_id"]) + 1

        # If we already have a script and append
        if append & (self.obs_wanted is not None):
            self.obs_wanted = np.concatenate([self.obs_wanted, obs_wanted])
            self.obs_wanted.sort(order=["mjd", "filter"])
        else:
            self.obs_wanted = obs_wanted

        self.mjd_start = self.obs_wanted["mjd"] - self.obs_wanted["mjd_tol"]
        # Here is the atribute that core scheduler checks to
        # broadcast scheduled observations in the conditions object.
        self.scheduled_obs = self.obs_wanted["mjd"]

    def _check_list(self, conditions):
        """Check to see if the current mjd is good"""
        observation = None
        if self.obs_wanted is not None:
            # Scheduled observations that are in the right
            # time window and have not been executed
            in_time_window = np.where(
                (self.mjd_start < conditions.mjd)
                & (self.obs_wanted["flush_by_mjd"] > conditions.mjd)
                & (~self.obs_wanted["observed"])
            )[0]

            if np.size(in_time_window) > 0:
                pass_checks = self._check_alts_ha(self.obs_wanted[in_time_window], conditions)
                matches = in_time_window[pass_checks]
            else:
                matches = []

            if np.size(matches) > 0:
                observation = self._slice2obs(self.obs_wanted[matches[0]])

        return observation

    def flush_script(self, conditions):
        """Remove things from the script that aren't needed anymore"""
        if self.obs_wanted is not None:
            still_relevant = np.where(
                (self.obs_wanted["observed"] == False) & (self.obs_wanted["flush_by_mjd"] > conditions.mjd)
            )[0]

            if np.size(still_relevant) > 0:
                observations = self.obs_wanted[still_relevant]
                self.set_script(observations, append=False)
            else:
                self.clear_script()

    def _spin_fields(self):
        """Spin the field tessellation to generate a random orientation

        The default field tesselation is rotated randomly in longitude,
        and then the pole is rotated to a random point on the sphere.

        Automatically advances self.spin_indx when called.

        """
        lon = self.lon[self.spin_indx]
        lat = self.lat[self.spin_indx]
        lon2 = self.lon2[self.spin_indx]

        # rotate longitude
        ra = (self.fields_init["RA"] + lon) % (2.0 * np.pi)
        dec = copy(self.fields_init["dec"])

        # Now to rotate ra and dec about the x-axis
        x, y, z = thetaphi2xyz(ra, dec + np.pi / 2.0)
        xp, yp, zp = rotx(lat, x, y, z)
        theta, phi = xyz2thetaphi(xp, yp, zp)
        dec = phi - np.pi / 2
        ra = theta + np.pi

        # One more RA rotation
        ra = (ra + lon2) % (2.0 * np.pi)

        self.fields["RA"] = ra
        self.fields["dec"] = dec
        # Rebuild the kdtree with the new positions
        # XXX-may be doing some ra,dec to conversions xyz more
        # than needed.
        self._hp2fieldsetup(ra, dec)

        # Advance the spin index
        self.spin_indx += 1

    def _tesselate(self, hpid_to_observe):
        self._spin_fields()
        field_ids = np.unique(self.hp2fields[hpid_to_observe])
        # Put the fields in a good order.
        better_order = order_observations(self.fields["RA"][field_ids], self.fields["dec"][field_ids])
        ras = self.fields["RA"][field_ids[better_order]]
        decs = self.fields["dec"][field_ids[better_order]]

        return ras, decs

    def _new_event(self, target_o_o, conditions):
        """A new ToO event, generate any observations for followup"""
        # flush out any old observations or ones that have been completed
        self.flush_script(conditions)

        # Check that this is the type of ToO we are supposed to trigger on
        correct_type = False
        for type_to_follow in self.too_types_to_follow:
            if type_to_follow in target_o_o.too_type:
                correct_type = True

        # Check that we have not seen this alert yet
        unseen = False
        if target_o_o.id not in self.seen_alerts:
            unseen = True

        if correct_type & unseen:
            self.seen_alerts.append(target_o_o.id)
            # Check that the event center is in the footprint
            # we want to observe
            hpid_center = _ra_dec2_hpid(self.nside, target_o_o.ra_rad_center, target_o_o.dec_rad_center)
            if self.followup_footprint[hpid_center] > 0:
                target_area = self.followup_footprint * target_o_o.footprint
                # generate a list of pointings for that area
                hpid_to_observe = np.where(target_area > 0)[0]
                if hpid_to_observe.size > 0:
                    ras, decs = self._tesselate(hpid_to_observe)
                else:
                    ras = np.array([target_o_o.ra_rad_center])
                    decs = np.array([target_o_o.dec_rad_center])

                # Figure out an MJD start time for the object
                # if it is still rising and low.
                alt, az = _approx_ra_dec2_alt_az(
                    target_o_o.ra_rad_center,
                    target_o_o.dec_rad_center,
                    conditions.site.latitude_rad,
                    None,
                    conditions.mjd,
                    lmst=np.max(conditions.lmst),
                )
                HA = np.max(conditions.lmst) - target_o_o.ra_rad_center * 12.0 / np.pi

                if (HA < self.HA_max) & (HA > self.HA_min):
                    t_to_rise = (self.HA_max - HA) / 24.0
                    mjd0 = conditions.mjd + t_to_rise
                else:
                    mjd0 = conditions.mjd + 0.0

                obs_list = []
                for time, filternames, nv, exptime, index in zip(
                    self.times,
                    self.filters_at_times,
                    self.nvis,
                    self.exptimes,
                    np.arange(np.size(self.times)),
                ):
                    for i in range(nv):
                        # let's dither each pointing
                        if (i != 0) & (hpid_to_observe.size > 0):
                            ras, decs = self._tesselate(hpid_to_observe)

                        for filtername in filternames:
                            # Subsitute y for z if needed on first observation
                            if i == 0:
                                if (filtername == "z") & (filtername not in conditions.mounted_filters):
                                    filtername = "y"

                            if filtername == "u":
                                nexp = self.n_usnaps
                            else:
                                nexp = self.n_snaps

                            # If we are doing a short exposure
                            # need to be 1 snap for shutter limits
                            if exptime < 29.0:
                                nexp = 1

                            # check if we should break
                            # long exposures into multiple
                            if self.split_long:
                                if exptime > 119:
                                    nexp = int(np.round(exptime / 30.0))

                            obs = ScheduledObservationArray(ras.size)
                            obs["RA"] = ras
                            obs["dec"] = decs
                            obs["mjd"] = mjd0 + time
                            obs["flush_by_mjd"] = mjd0 + time + self.flushtime
                            obs["exptime"] = exptime
                            obs["nexp"] = nexp
                            obs["filter"] = filtername
                            obs["rotSkyPos"] = 0  # XXX--maybe throw a rotation detailer in here
                            obs["mjd_tol"] = self.mjd_tol
                            obs["dist_tol"] = self.dist_tol
                            obs["alt_min"] = self.alt_min
                            obs["alt_max"] = self.alt_max
                            obs["HA_max"] = self.HA_max
                            obs["HA_min"] = self.HA_min

                            obs["scheduler_note"] = self.survey_name + ", %i_t%i_i%i" % (
                                target_o_o.id,
                                time * 24,
                                index,
                            )
                            obs_list.append(obs)
                observations = np.concatenate(obs_list)
                self.set_script(observations)

    def calc_reward_function(self, conditions):
        """If there is an observation ready to go, execute it,
        otherwise, -inf"""
        # check if any new event has come in

        if conditions.targets_of_opportunity is not None:
            for target_o_o in conditions.targets_of_opportunity:
                if target_o_o.id > self.last_event_id:
                    self._new_event(target_o_o, conditions)
                    self.last_event_id = target_o_o.id

        observation = self.generate_observations_rough(conditions)

        if (observation is None) | (len(observation) == 0):
            self.reward = -np.inf
        else:
            self.reward = self.reward_val
        return self.reward

    def generate_observations(self, conditions):
        observations = self.generate_observations_rough(conditions)

        if len(observations) > 0:
            for detailer in self.detailers:
                observations = detailer(observations, conditions)

        return observations


def mean_longitude(longitude):
    """Compute a mean longitude, accounting for wrap around."""
    x = np.cos(longitude)
    y = np.sin(longitude)
    meanx = np.mean(x)
    meany = np.mean(y)
    angle = np.arctan2(meany, meanx)
    radius = np.sqrt(meanx**2 + meany**2)
    mid_longitude = angle % (2.0 * np.pi)
    if radius < 0.1:
        mid_longitude = np.pi
    return mid_longitude


def gen_too_surveys(
    nside=32,
    detailer_list=None,
    too_footprint=None,
    split_long=False,
    n_snaps=2,
):
    result = []

    ############
    # Generic GW followup
    ############

    # XXX---There's some extra stuff about do different things
    # if there's limited time before it sets. Let's see if this can
    # work first

    # XXX--instructions say do 4th night only 1/3 of the time.
    # Just leaving off for now

    times = [0, 24, 48]
    filters_at_times = ["ugrizy", "ugrizy", "ugrizy"]
    nvis = [3, 1, 1]
    exptimes = [120.0, 120.0, 120.0]
    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["GW_case_A"],
            survey_name="ToO, GW_case_A",
            split_long=split_long,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    times = [0, 24, 48, 72]
    filters_at_times = ["gri", "ri", "ri", "ri"]
    nvis = [3, 1, 1, 1]
    exptimes = [120.0, 180.0, 180.0, 180.0]
    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["GW_case_B", "GW_case_C"],
            survey_name="ToO, GW_case_B_C",
            split_long=split_long,
            flushtime=48,
            n_snaps=n_snaps,
        )
    )

    times = [0, 24, 48, 72]
    filters_at_times = ["gr", "gr", "gr", "gr"]
    nvis = [1, 1, 1, 1]
    exptimes = [120.0, 120.0, 120.0, 120.0]
    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["GW_case_D", "GW_case_E"],
            survey_name="ToO, GW_case_D_E",
            split_long=split_long,
            flushtime=48,
            n_snaps=n_snaps,
        )
    )

    ############
    # Black hole-black hole GW merger
    ############

    # XXX--only considering bright objects now.

    times = np.array([0, 2, 7, 9, 39]) * 24
    filters_at_times = ["ugri"] * 5
    nvis = [1] * 5
    exptimes = [DEFAULT_EXP_TIME] * 5

    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["BBH_case_A", "BBH_case_B", "BBH_case_C"],
            survey_name="ToO, BBH",
            split_long=split_long,
            flushtime=48,
            n_snaps=n_snaps,
        )
    )

    ############
    # Lensed BNS
    ############

    times = [1.0, 1.0]
    filters_at_times = ["g", "r"]
    nvis = [1, 1]
    exptimes = [DEFAULT_EXP_TIME, 90.0]

    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["lensed_BNS_case_A"],
            survey_name="ToO, LensedBNS_A",
            split_long=split_long,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    times = [1.0]
    filters_at_times = [
        "gr",
    ]
    nvis = [10]
    exptimes = [150]

    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["lensed_BNS_case_B"],
            survey_name="ToO, LensedBNS_B",
            split_long=split_long,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    ############
    # Neutrino detector followup
    ############

    # XXX--need to update footprint to cut out galactic latitude

    times = [0, 0, 15 / 60.0, 0.5, 24, 24.5, 144]
    filters_at_times = ["g", "r", "z", "g", "r", "z", "grz"]
    exptimes = [
        120,
        DEFAULT_EXP_TIME,
        DEFAULT_EXP_TIME,
        120,
        DEFAULT_EXP_TIME,
        DEFAULT_EXP_TIME,
        DEFAULT_EXP_TIME,
    ]
    nvis = [1, 1, 1, 1, 1, 1, 1]

    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["neutrino"],
            survey_name="ToO, neutrino",
            split_long=split_long,
            flushtime=8.0,
            n_snaps=n_snaps,
        )
    )

    times = [0]
    filters_at_times = ["u"]
    exptimes = [DEFAULT_EXP_TIME]
    nvis = [1]

    # U-band with very long flush time.

    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["neutrino"],
            survey_name="ToO, neutrino_u",
            split_long=split_long,
            flushtime=1440,
            n_snaps=n_snaps,
        )
    )

    ############
    # Solar System
    ############
    # For the solar system objects, probably want a custom survey object,
    # but this should work for now. Want to add a detailer to add a dither
    # position.

    times = [0, 33 / 60.0, 66 / 60.0]
    filters_at_times = ["r"] * 3
    nvis = [1] * 3
    exptimes = [DEFAULT_EXP_TIME] * 3

    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["SSO_night"],
            survey_name="ToO, SSO_night",
            split_long=split_long,
            flushtime=2.0,
            n_snaps=n_snaps,
        )
    )

    times = [0, 10 / 60.0, 20 / 60.0]
    filters_at_times = ["z"] * 3
    nvis = [2] * 3
    exptimes = [15.0] * 3

    result.append(
        ToOScriptedSurvey(
            [],
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            filters_at_times=filters_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=detailer_list,
            too_types_to_follow=["SSO_twilight"],
            survey_name="ToO, SSO_twi",
            split_long=split_long,
            flushtime=2.0,
            n_snaps=n_snaps,
        )
    )

    return result
