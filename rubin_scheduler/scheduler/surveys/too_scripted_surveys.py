__all__ = ("ToOScriptedSurvey", "gen_too_surveys")

import warnings
from copy import copy, deepcopy

import healpy as hp
import numpy as np

import rubin_scheduler.scheduler.basis_functions as basis_functions
from rubin_scheduler.scheduler.detailers import BandPickToODetailer, TrackingInfoDetailer
from rubin_scheduler.scheduler.surveys import BaseMarkovSurvey, ScriptedSurvey
from rubin_scheduler.scheduler.utils import (
    ScheduledObservationArray,
    comcam_tessellate,
    order_observations,
    thetaphi2xyz,
    xyz2thetaphi,
)
from rubin_scheduler.site_models import _read_fields
from rubin_scheduler.utils import (
    DEFAULT_NSIDE,
    _approx_ra_dec2_alt_az,
    _build_tree,
    _hpid2_ra_dec,
    _ra_dec2_hpid,
)


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
    bands_at_times : list of str
        The bands that should be observed at each time in `times`.
        Default ["gz", "gz", "gz", "gz", "gz", "gz"]
    nvis : list of int
        The number of visits per band at each time in `times`.
        Default [1, 1, 1, 1, 6, 6]
    exptimes : list of floats
        The exposure times to use for each time in `times`.
        Default [DEFAULT_EXP_TIME]*5,
    alt_min : float
        Do not attempt observations below this limit (degrees).
        Note the telescope alt limit is 20 degrees, however, slew
        and band change time means alt_min here should be set higher
        (otherwise, target will pass altitude check, but then fail
        to observe by the time the telescope gets there).
    target_name_base : `str`
        String to use as the base of the target name. Will be appended
        with an integer for the object id.
    split_long : `bool`
        Should long exposure times be split into multiple snaps.
        Default False.
    split_long_max : `float`
        Maximum exposure time to allow before splitting into
        multiple snaps if split_long is True. Default 30s.
    split_long_div : `float`
        Time to divide the exposure time by to decide how many
        snaps to use. Default 60s.
    event_gen_detailers : `list`
        A list of detailers to run on arrays right after generating
        a list.
    simple_single_tesselate : `bool`
        If tesselating a single HEALpixel, use the center of the
        HEALpixel for the pointing rather than try to look up
        pointings that cover the center and all corners. Default True.
    dither_per_visit : `bool`
        If nvis > 1, randomize the tesselation between each visit.
        Probably what you want to do if you're searching for an event
        and worried about chip and raft gaps. Default True.
    """

    def __init__(
        self,
        basis_functions,
        followup_footprint=None,
        nside=DEFAULT_NSIDE,
        reward_val=1e6,
        times=[1, 2, 4, 8, 24, 48],
        bands_at_times=["gz", "gz", "gz", "gz", "gz", "gz"],
        nvis=[1, 1, 1, 1, 6, 6],
        exptimes=[DEFAULT_EXP_TIME] * 6,
        camera="LSST",
        survey_name="ToO",
        target_name_base="ToO",
        observation_reason="ToO",
        science_program=None,
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
        split_long_max=30.0,
        split_long_div=60.0,
        filters_at_times=None,
        event_gen_detailers=None,
        return_n_limit=500,
        simple_single_tesselate=True,
        dither_per_visit=True,
    ):
        if filters_at_times is not None:
            warnings.warn("filters_at_times deprecated in favor of bands_at_times", FutureWarning)
            bands_at_times = filters_at_times
        # Make sure lists all match
        check = np.unique([len(bands_at_times), len(times), len(nvis), len(exptimes)])
        if np.size(check) > 1:
            raise ValueError("lengths of times, bands_at_times, nvis, and exptimes must match.")

        # Figure out what else I need to super here

        if ignore_obs is None:
            ignore_obs = []

        self.basis_functions = basis_functions
        self.basis_weights = [0] * len(basis_functions)
        self.survey_name = survey_name
        self.followup_footprint = followup_footprint
        self.last_event_id = -1
        self.night = -1
        self.reward_val = reward_val
        self.times = np.array(times) / 24.0  # to days
        self.bands_at_times = bands_at_times
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
        self.return_n_limit = return_n_limit
        self.extra_features = {}
        self.extra_basis_functions = {}
        self.simple_single_tesselate = simple_single_tesselate
        self.dither_per_visit = dither_per_visit
        if detailers is None:
            self.detailers = []
        else:
            self.detailers = detailers
        if event_gen_detailers is None:
            self.event_gen_detailers = []
        else:
            self.event_gen_detailers = event_gen_detailers
        self.dither = dither
        self.id_start = id_start
        self.last_mjd = -1
        self.too_types_to_follow = too_types_to_follow
        self.split_long = split_long
        self.target_name_base = target_name_base
        self.split_long_max = split_long_max
        self.split_long_div = split_long_div

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

        self._hp2fieldsetup(self.fields["RA"], self.fields["dec"])
        self._hpcorners2fieldsetup(self.fields["RA"], self.fields["dec"])

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
        # Check if there is a detailer for tracking info
        has_tracking_detailer = False
        for detailer in self.detailers:
            if isinstance(detailer, TrackingInfoDetailer):
                has_tracking_detailer = True
        # Add information for visit metadata if necessary
        if (science_program is not None) | (observation_reason is not None):
            should_have_tracking_detailer = True
        else:
            should_have_tracking_detailer = False
        if should_have_tracking_detailer:
            # Check if one already present - will use that if so.
            if has_tracking_detailer:
                warnings.warn(
                    f"Survey {self.survey_name} has a tracking detailer but "
                    f"observation_reason or science_program also set (and will ignore)."
                )
            else:
                self.detailers.append(
                    TrackingInfoDetailer(
                        science_program=science_program,
                        observation_reason=observation_reason,
                    )
                )

    def flush_script(self, conditions):
        """Remove things from the script that aren't needed anymore"""
        if self.obs_wanted is not None:
            still_relevant = np.where(
                np.logical_not(self.obs_wanted["observed"])
                & (self.obs_wanted["flush_by_mjd"] > conditions.mjd)
            )[0]

            if np.size(still_relevant) > 0:
                observations = self.obs_wanted[still_relevant]
                self.set_script(observations, append=False, add_index=False)
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
        self._hpcorners2fieldsetup(ra, dec)

        # Advance the spin index
        self.spin_indx += 1

    def _hpcorners2fieldsetup(self, ra, dec):
        """Map each healpixel corner to nearest field.
        Parameters
        ----------
        ra : `float`
            The RA of the possible pointings (radians)
        dec : `float`
            The decs of the possible pointings (radians)
        """
        hpid = np.arange(hp.nside2npix(self.nside))
        xyz = hp.boundaries(self.nside, hpid)
        tree = _build_tree(ra, dec)

        self.hpcorners2fields = []
        for i in range(4):
            dist, ind = tree.query(np.vstack([xyz[:, 0, i], xyz[:, 1, i], xyz[:, 2, i]]).T, k=1)
            self.hpcorners2fields.append(ind)

    def _tesselate(self, hpid_to_observe):

        if self.simple_single_tesselate & (len(hpid_to_observe) == 1):
            ras, decs = _hpid2_ra_dec(self.nside, hpid_to_observe)
        else:
            self._spin_fields()
            # Closest pointings to all the HEALpix centers
            center_field_ids = np.unique(self.hp2fields[hpid_to_observe])
            # Closest pointings to all the HEALpix corners
            corners = np.concatenate([np.unique(ind[hpid_to_observe]) for ind in self.hpcorners2fields])
            field_ids = np.unique(np.concatenate([center_field_ids, corners]))
            # Put the fields in a good order
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
                # ToO footprint could be any nside, so match here
                # hp.ud_grade takes a mean. So a 1/0 mask will be forgiving
                # and expand, but a mask using np.nan will be hard.
                matched_target_o_o_fp = hp.ud_grade(target_o_o.footprint, self.nside)
                target_area = self.followup_footprint * matched_target_o_o_fp
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
                for time, bandnames, nv, exptime, index in zip(
                    self.times,
                    self.bands_at_times,
                    self.nvis,
                    self.exptimes,
                    np.arange(np.size(self.times)),
                ):
                    # Could potentially throw in a dither change here
                    # so it is different at each time as well?
                    for i in range(nv):
                        # let's dither each pointing
                        if (i != 0) & (hpid_to_observe.size > 0) & (self.dither_per_visit):
                            ras, decs = self._tesselate(hpid_to_observe)

                        for bandname in bandnames:
                            # Subsitute y for z if needed on first observation
                            if i == 0:
                                if (bandname == "z") & (bandname not in conditions.mounted_bands):
                                    bandname = "y"

                            if bandname == "u":
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
                                if exptime > self.split_long_max:
                                    nexp = int(np.ceil(exptime / self.split_long_div))

                            obs = ScheduledObservationArray(ras.size)
                            obs["RA"] = ras
                            obs["dec"] = decs
                            obs["mjd"] = mjd0 + time
                            obs["flush_by_mjd"] = mjd0 + time + self.flushtime
                            obs["exptime"] = exptime
                            obs["nexp"] = nexp
                            obs["band"] = bandname
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
                            obs["target_name"] = self.target_name_base + "_i%i" % index
                            obs_list.append(obs)
                observations = np.concatenate(obs_list)
                for detailer in self.event_gen_detailers:
                    observations = detailer(observations, conditions, target_o_o=target_o_o)
                self.set_script(observations)

    def update_conditions(self, conditions):
        if conditions.targets_of_opportunity is not None:
            for target_o_o in conditions.targets_of_opportunity:
                if target_o_o.id > self.last_event_id:
                    self._new_event(target_o_o, conditions)
                    self.last_event_id = target_o_o.id

    def calc_reward_function(self, conditions):
        """If there is an observation ready to go, execute it,
        otherwise, -inf"""

        # check if any new event has come in
        self.update_conditions(conditions)

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
    nside=DEFAULT_NSIDE,
    detailer_list=None,
    too_footprint=None,
    split_long=False,
    long_exp_nsnaps=2,
    n_snaps=2,
    wind_speed_maximum=20.0,
    observation_reason="ToO",
    science_program=None,
):

    warnings.warn("Function gen_too_surveys moving out of rubin_scheduler.", DeprecationWarning)
    result = []
    bf_list = []
    bf_list.append(basis_functions.AvoidDirectWind(wind_speed_maximum=wind_speed_maximum, nside=nside))
    bf_list.append(basis_functions.MoonAvoidanceBasisFunction(moon_distance=30.0))
    ############
    # Generic GW followup
    ############

    # XXX---There's some extra stuff about do different things
    # if there's limited time before it sets. Let's see if this can
    # work first

    # XXX--instructions say do 4th night only 1/3 of the time.
    # Just leaving off for now

    times = [0, 24, 48]
    bands_at_times = ["ugrizy", "ugrizy", "ugrizy"]
    nvis = [3, 1, 1]
    exptimes = [120.0, 120.0, 120.0]
    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_A"],
            survey_name="ToO, GW_case_A",
            split_long=split_long,
            flushtime=48.0,
            n_snaps=long_exp_nsnaps,
            target_name_base="GW_case_A",
            observation_reason=observation_reason,
            science_program=science_program,
        )
    )

    ############
    # GW gold and GW unidentified gold
    ############

    times = [0, 2, 4, 24, 48, 72]
    bands_at_times = ["gri", "gri", "gri", "ri", "ri", "ri"]
    nvis = [4, 4, 4, 6, 6, 6]
    exptimes = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_B", "GW_case_C"],
            survey_name="ToO, GW_case_B_C",
            target_name_base="GW_case_B_C",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48,
            n_snaps=long_exp_nsnaps,
        )
    )

    ############
    # GW silver and GW unidentified silver
    ############

    times = [0, 24, 48, 72]
    bands_at_times = ["gi", "gi", "gi", "gi"]
    nvis = [1, 4, 4, 4]
    exptimes = [30.0, 30.0, 30.0, 30.0]
    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["GW_case_D", "GW_case_E"],
            survey_name="ToO, GW_case_D_E",
            target_name_base="GW_case_D_E",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48,
            n_snaps=long_exp_nsnaps,
            event_gen_detailers=None,
        )
    )

    ############
    # BBH hole-black hole GW merger
    # If nearby (dist < 2200 Mpc) and dark time, use ugi
    # If distant (dist > 2200 Mpc) and dark time, use rgi
    # If bright time, use rzi
    ############

    # XXX--only considering bright objects now.

    # SM-- adding support for different BBH cases. We should have
    # a discussion about how to differentiate these, and if it is
    # possible to do so through the alert stream.

    event_detailers = [
        BandPickToODetailer(
            band_start="z", band_end="g", distance_limit=30e10, check_mounted=True, require_dark=True
        ),
        BandPickToODetailer(
            band_start="r", band_end="u", distance_limit=2.2e6, check_mounted=True, require_dark=True
        ),
    ]
    times = np.array([0, 2, 7, 9, 39]) * 24
    bands_at_times = ["rzi"] * times.size
    nvis = [1] * times.size
    exptimes = [DEFAULT_EXP_TIME] * times.size

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["BBH_case_A", "BBH_case_B", "BBH_case_C"],
            survey_name="ToO, BBH",
            target_name_base="BBH",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48,
            n_snaps=n_snaps,
            event_gen_detailers=event_detailers,
        )
    )

    ############
    # Lensed BNS
    ############

    times = np.array([1.0, 1.0, 25, 25, 49, 49])
    bands_at_times = ["g", "r"] * 3
    nvis = [1, 3] * 3
    exptimes = [DEFAULT_EXP_TIME, DEFAULT_EXP_TIME] * 3

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["lensed_BNS_case_A"],
            survey_name="ToO, LensedBNS_A",
            target_name_base="LensedBNS_A",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    # This is the small skymap (15 deg^2 case)
    times = np.array([1.0, 1.0, 25, 25, 49, 49])
    bands_at_times = ["g", "r"] * 3
    nvis = [180, 120] * 3
    exptimes = [30] * times.size

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["lensed_BNS_case_B"],
            survey_name="ToO, LensedBNS_B",
            target_name_base="LensedBNS_B",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48.0,
            n_snaps=long_exp_nsnaps,
        )
    )

    ############
    # Neutrino detector followup
    ############

    times = [0, 0, 15 / 60.0, 0, 24, 24, 144, 144]
    bands_at_times = ["u", "g", "r", "z", "g", "r", "g", "rz"]
    exptimes = [
        30,
        30,
        DEFAULT_EXP_TIME,
        DEFAULT_EXP_TIME,
        30,
        DEFAULT_EXP_TIME,
        30,
        DEFAULT_EXP_TIME,
    ]
    nvis = [1, 4, 1, 1, 4, 1, 4, 1]

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["neutrino"],
            survey_name="ToO, neutrino",
            target_name_base="neutrino",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=20 * 24,
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
    bands_at_times = ["r"] * 3
    nvis = [1] * 3
    exptimes = [DEFAULT_EXP_TIME] * 3

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SSO_night"],
            survey_name="ToO, SSO_night",
            target_name_base="SSO_night",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=3.0,
            n_snaps=n_snaps,
        )
    )

    times = [0, 10 / 60.0, 20 / 60.0]
    bands_at_times = ["z"] * 3
    nvis = [2] * 3
    exptimes = [15.0] * 3

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SSO_twilight"],
            survey_name="ToO, SSO_twi",
            target_name_base="SSO_twi",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=3.0,
            n_snaps=n_snaps,
        )
    )

    ############
    # Galactic Supernova
    ############
    # For galactic supernovae, we want to tile continuously
    # in the region in 1s and 15s exposures for i band, until
    # a counterpart is identified

    times = [0, 0, 0, 0] * 4
    bands_at_times = ["i", "i", "i", "i"] * 4
    nvis = [1, 1, 1, 1] * 4
    exptimes = [1, 15, 1, 15] * 4

    result.append(
        ToOScriptedSurvey(
            bf_list,
            nside=nside,
            followup_footprint=too_footprint,
            times=times,
            bands_at_times=bands_at_times,
            nvis=nvis,
            exptimes=exptimes,
            detailers=deepcopy(detailer_list),
            too_types_to_follow=["SN_Galactic"],
            survey_name="ToO, galactic SN",
            target_name_base="SN_Galactic",
            observation_reason=observation_reason,
            science_program=science_program,
            split_long=split_long,
            flushtime=48.0,
            n_snaps=n_snaps,
        )
    )

    return result
