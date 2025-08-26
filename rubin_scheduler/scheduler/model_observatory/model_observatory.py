__all__ = ("ModelObservatory",)

import copy
import warnings

import healpy as hp
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

import rubin_scheduler.skybrightness_pre as sb
from rubin_scheduler.data import data_versions
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import KinemModel
from rubin_scheduler.scheduler.utils import smallest_signed_angle

# For backwards compatibility
from rubin_scheduler.site_models import (
    Almanac,
    CloudData,
    ConstantCloudData,
    ConstantSeeingData,
    ScheduledDowntimeData,
    SeeingData,
    SeeingModel,
    UnscheduledDowntimeMoreY1Data,
)
from rubin_scheduler.utils import (
    DEFAULT_NSIDE,
    SURVEY_START_MJD,
    Site,
    _angular_separation,
    _approx_altaz2pa,
    _approx_ra_dec2_alt_az,
    _hpid2_ra_dec,
    _ra_dec2_hpid,
    calc_lmst,
    calc_season,
    m5_flat_sed,
    rotation_converter,
)


class ModelObservatory:
    """Generate a realistic telemetry stream for the scheduler in simulations,
    including simulating the acquisition of observations.

    Parameters
    ----------
    nside : `int`, optional
        The healpix nside resolution.
        Default None uses `set_default_nside()`.
    mjd : `float`, optional
        The MJD to start the model observatory for observations.
        Used to set current conditions and load sky.
        Default None uses mjd_start.
    mjd_start : `float`, optional
        The MJD of the start of the survey.
        This must be set to start of whole survey, for tracking
        purposes. Should be during the day so night will be
        computed properly.
    alt_min : `float`, optional
        The minimum altitude to compute models at (degrees).
    lax_dome : `bool`, optional
        Passed to observatory model. If true, allows dome creep.
    cloud_limit : `float`, optional.
        The limit to stop taking observations if the cloud model returns
        something equal or higher. Default of 0.3 is validated as a
        "somewhat pessimistic" weather downtime choice.
    sim_to_o : `sim_targetoO` object, optional
        If one would like to inject simulated ToOs into the telemetry
        stream. Default None adds no ToOs.
    park_after : `float`, optional
        Park the telescope after a gap longer than park_after (minutes).
        Default 10 minutes is used to park the telescope during downtime.
    init_load_length : `int`, optional
        The length of pre-scheduled sky brightness values to load
        initially (days). The default is 10 days; shorter values
        can be used for quicker load times.
    kinem_model : `~.scheduler.model_observatory.Kinem_model`, optional
        An instantiated rubin_scheduler Kinem_model object.
        Default of None uses a default Kinem_model.
    seeing_db : `str`, optional
        The filename of the seeing data database, if one would like
        to use an alternate seeing database.
        Default None uses the default seeing database.
    seeing_data : `~.site_models.SeeingData`-like, optional
        If one wants to replace the default seeing_data object.
        Should be an object with a
        __call__ method that takes MJD and returns zenith fwhm_500 in
        arcsec. Set to "ideal" to have constant 0.7" seeing.
    cloud_db : `str`, optional
        The filename of the cloud data database.
        Default of None uses the default database from rubin_sim_data.
    cloud_offset_year : `int`, optional
        The year offset to be passed to CloudData. Default 0.
    cloud_data : `~.site_models.CloudData`-like, optional
        If one wants to replace the default cloud data. Should be an
        object with a __call__ method that takes MJD and returns
        cloudy level. Set to "ideal" for  no clouds.
    downtimes : `np.ndarray`, (N,3) or None, optional
        If one wants to replace the default downtimes. Should be a
        np.array with columns of "start" and "end" with MJD values
        and should include both scheduled and unscheduled downtime.
        Set to "ideal" for no downtime. Default of None will use
        the downtime models from `~.site_models.ScheduledDowntime`
        and `~.site_models.UnscheduledDowntime`.
    no_sky : `bool`, optional
        If True, then don't load any skybrightness files.
        Handy if one wants a well filled out Conditions object,
        but doesn't need the sky since that can be slower to load.
        Default False.
    wind_data : ~.site_models.WindData`-like, optional
        If one wants to replace the default wind_data object. Should
        be an object with a __call__ method that takes the time and
        returns a tuple with the wind speed (m/s) and originating
        direction (radians east of north).
        Default of None uses an idealized WindData object with no wind.
    starting_time_key : `str`, optional
        What key in the almanac to use to determine the start of
        observing on a night. Default "sun_n12_setting", e.g., sun
        at -12 degrees and setting. Other options are
        "sun_n18_setting" and "sunset".
        If surveys are not configured to wait until the sun is lower
        in the sky, observing will start as soon as the time passes
        the time returned by this key, in each night.
    ending_time_key : `str`, optional
        What key in the almanac to use to signify it is time to skip
        to the next night. Default "sun_n12_rising", e.g., sun at
        -12 degrees and rising. Other options are "sun_n18_rising"
        and "sunrise".
    sky_az_limits : `list` [[`float`, `float`]]
        A list of lists giving valid azimuth ranges. e.g.,
        [0, 180] would mean all azimuth values are valid, while
        [[0, 90], [270, 360]] or [270, 90] would mean anywhere in
        the south is invalid.  Degrees.
    sky_alt_limits : `list` [[`float`, `float`]]
        A list of lists giving valid altitude ranges. Degrees.
        For both the alt and az limits, if a particular alt (or az)
        value is included in any limit, it is valid for all of them.
        Altitude limits of [[20, 40], [40, 60]] would allow altitudes
        betwen 20-40 and 40-60, but [[20, 40], [40, 60], [20, 86]]
        will allow altitudes anywhere between 20-86 degrees.
    telescope : `str`
        Telescope name for rotation computations. Default "rubin".
    resolve_rotskypos : `bool`
        If a requested value of rotSkyPos is not accessible, try adding
        90 degrees until it works.

    """

    def __init__(
        self,
        nside=DEFAULT_NSIDE,
        mjd=None,
        mjd_start=SURVEY_START_MJD,
        alt_min=5.0,
        lax_dome=True,
        cloud_limit=0.3,
        sim_to_o=None,
        park_after=10.0,
        init_load_length=10,
        kinem_model=None,
        seeing_db=None,
        seeing_data=None,
        cloud_db=None,
        cloud_offset_year=0,
        cloud_data=None,
        downtimes=None,
        no_sky=False,
        wind_data=None,
        starting_time_key="sun_n12_setting",
        ending_time_key="sun_n12_rising",
        sky_alt_limits=None,
        sky_az_limits=None,
        telescope="rubin",
        cloud_maps=None,
        resolve_rotskypos=True,
    ):
        self.nside = nside
        self.resolve_rotskypos = resolve_rotskypos

        # Set the time now - mjd
        # and the time of the survey start
        self.mjd_start = mjd_start

        if mjd is None:
            mjd = mjd_start

        self.bandlist = ["u", "g", "r", "i", "z", "y"]

        self.cloud_limit = cloud_limit
        self.no_sky = no_sky
        self.alt_min = np.radians(alt_min)
        self.lax_dome = lax_dome
        self.starting_time_key = starting_time_key
        self.ending_time_key = ending_time_key

        self.sim__to_o = sim_to_o

        self.park_after = park_after / 60.0 / 24.0  # To days

        # Rotation converter
        self.rc = rotation_converter(telescope=telescope)

        # Create an astropy location
        self.site = Site("LSST")
        self.location = EarthLocation(
            lat=self.site.latitude, lon=self.site.longitude, height=self.site.height
        )

        # Set up the almanac - use mjd_start to keep "night" count the same.
        self.almanac = Almanac(mjd_start=self.mjd_start)

        # Load up all the models we need
        # Use mjd_start to ensure models always initialize to the same
        # starting point in time.
        mjd_start_time = Time(self.mjd_start, format="mjd")

        # Set up the downtime
        if isinstance(downtimes, str):
            if downtimes == "ideal":
                self.downtimes = np.array(
                    list(zip([], [])),
                    dtype=list(zip(["start", "end"], [float, float])),
                )
            else:
                warnings.warn("Downtimes should be a string equal to " "'ideal', an array or None")
        elif downtimes is None:
            self.down_nights = []
            self.sched_downtime_data = ScheduledDowntimeData(mjd_start_time)
            self.unsched_downtime_data = UnscheduledDowntimeMoreY1Data(mjd_start_time)

            sched_downtimes = self.sched_downtime_data()
            unsched_downtimes = self.unsched_downtime_data()

            down_starts = []
            down_ends = []
            for dt in sched_downtimes:
                down_starts.append(dt["start"].mjd)
                down_ends.append(dt["end"].mjd)
            for dt in unsched_downtimes:
                down_starts.append(dt["start"].mjd)
                down_ends.append(dt["end"].mjd)

            self.downtimes = np.array(
                list(zip(down_starts, down_ends)),
                dtype=list(zip(["start", "end"], [float, float])),
            )
            self.downtimes.sort(order="start")

            # Make sure there aren't any overlapping downtimes
            diff = self.downtimes["start"][1:] - self.downtimes["end"][0:-1]
            while np.min(diff) < 0:
                # Should be able to do this without a loop, but this works
                for i, dt in enumerate(self.downtimes[0:-1]):
                    if self.downtimes["start"][i + 1] < dt["end"]:
                        new_end = np.max([dt["end"], self.downtimes["end"][i + 1]])
                        self.downtimes[i]["end"] = new_end
                        self.downtimes[i + 1]["end"] = new_end

                good = np.where(self.downtimes["end"] - np.roll(self.downtimes["end"], 1) != 0)
                self.downtimes = self.downtimes[good]
                diff = self.downtimes["start"][1:] - self.downtimes["end"][0:-1]
        else:
            self.downtimes = downtimes

        # Set the wind data
        self.wind_data = wind_data

        # Set up the seeing data
        if seeing_data == "ideal":
            self.seeing_data = ConstantSeeingData()
        elif seeing_data is not None:
            self.seeing_data = seeing_data
        else:
            self.seeing_data = SeeingData(mjd_start_time, seeing_db=seeing_db)
        self.seeing_model = SeeingModel()
        self.seeing_indx_dict = {}
        for i, bandname in enumerate(self.seeing_model.band_list):
            self.seeing_indx_dict[bandname] = i

        self.seeing_fwhm_eff = {}
        for key in self.bandlist:
            self.seeing_fwhm_eff[key] = np.zeros(hp.nside2npix(self.nside), dtype=float)

        # Set up the cloud data
        if cloud_data == "ideal":
            self.cloud_data = ConstantCloudData()
        elif cloud_data is not None:
            self.cloud_data = cloud_data
        else:
            self.cloud_data = CloudData(mjd_start_time, cloud_db=cloud_db, offset_year=cloud_offset_year)

        # Set up the skybrightness
        if not self.no_sky:
            self.sky_model = sb.SkyModelPre(init_load_length=init_load_length)
        else:
            self.sky_model = None

        # Set up the kinematic model
        if kinem_model is None:
            self.observatory = KinemModel(mjd0=self.mjd_start)
        else:
            self.observatory = kinem_model

        # Pick up tel alt and az limits from kwargs
        if sky_alt_limits is not None:
            self.sky_alt_limits = np.radians(sky_alt_limits)
        else:
            self.sky_alt_limits = None
        if sky_az_limits is not None:
            self.sky_az_limits = np.radians(sky_az_limits)
        else:
            self.sky_az_limits = None
        # Add the observatory alt/az limits to the user-defined limits
        # But we do have to be careful that we're not overriding more
        # restrictive limits that were already set.
        # So we'll just keep these separate.
        self.tel_alt_limits = copy.deepcopy(
            [
                self.observatory.telalt_minpos_rad,
                self.observatory.telalt_maxpos_rad,
            ]
        )
        self.tel_az_limits = copy.deepcopy(
            [self.observatory.telaz_minpos_rad, self.observatory.telaz_maxpos_rad]
        )
        # Each of these limits will be treated as hard limits that we don't
        # want pointings to stray into, so add a pad around the values
        self.altaz_limit_pad = np.radians(2.0)

        # Let's make sure we're at an openable MJD
        good_mjd = False
        to_set_mjd = mjd
        while not good_mjd:
            good_mjd, to_set_mjd = self.check_mjd(to_set_mjd)
        self.mjd = to_set_mjd

        # Create the map of the season offsets - this map is constant
        ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(self.nside)))
        ra_deg = np.degrees(ra)
        self.season_map = calc_season(ra_deg, [self.mjd_start], self.mjd_start).flatten()
        # Set the sun_ra_start information, for the rolling footprints
        sun_moon_info = self.almanac.get_sun_moon_positions(self.mjd_start)
        self.sun_ra_start = sun_moon_info["sun_RA"] + 0
        # Conditions object to update and return on request
        # (at present, this is not updated -- recreated, below).
        self.conditions = Conditions(
            nside=self.nside,
        )

        self.obs_id_counter = 0

        self.cloud_maps = cloud_maps

    def get_info(self):
        """
        Returns
        -------
        Array with model versions that were instantiated
        """

        # Could add in the data version
        result = []
        versions = data_versions()
        for key in versions:
            result.append([key, versions[key]])

        return result

    def return_conditions(self):
        """
        Returns
        -------
        rubin_scheduler.scheduler.features.conditions object
        """
        self.conditions = Conditions(
            nside=self.nside,
            mjd=self.mjd,
        )

        # Current time as astropy time
        current_time = Time(self.mjd, format="mjd")

        # Clouds. XXX--just the raw value
        self.conditions.bulk_cloud = self.cloud_data(current_time)

        # use conditions object itself to get approx altitude of each
        # healpx
        alts = self.conditions.alt
        azs = self.conditions.az

        good = np.where(alts > self.alt_min)

        # reset the seeing
        for key in self.seeing_fwhm_eff:
            self.seeing_fwhm_eff[key].fill(np.nan)
        # Use the model to get the seeing at this time and airmasses.
        fwhm_500 = self.seeing_data(current_time)
        self.fwhm_500 = fwhm_500
        seeing_dict = self.seeing_model(fwhm_500, self.conditions.airmass[good])
        fwhm_eff = seeing_dict["fwhmEff"]
        for i, key in enumerate(self.seeing_model.band_list):
            self.seeing_fwhm_eff[key][good] = fwhm_eff[i, :]
        self.conditions.fwhm_eff = self.seeing_fwhm_eff

        # sky brightness
        if self.sky_model is not None:
            self.conditions.skybrightness = self.sky_model.return_mags(self.mjd)

        # Model observatory can continue to only refer to band and not filter
        # However, just to ensure set_auxtel_info and set_maintel_info
        # have something to pass in for filters (if those calls being used):
        self.conditions.mounted_filters = self.observatory.mounted_bands
        self.conditions.current_filter = self.observatory.current_band[0]
        # And technically, these next lines are unnecessary
        self.conditions.mounted_bands = self.observatory.mounted_bands
        self.conditions.current_band = self.observatory.current_band[0]

        # Compute the slewtimes
        slewtimes = np.empty(alts.size, dtype=float)
        slewtimes.fill(np.nan)
        # If there has been a gap, park the telescope
        gap = self.mjd - self.observatory.last_mjd
        if gap > self.park_after:
            self.observatory.park()
        slewtimes[good] = self.observatory.slew_times(
            0.0,
            0.0,
            self.mjd,
            alt_rad=alts[good],
            az_rad=azs[good],
            bandname=self.observatory.current_band,
            lax_dome=self.lax_dome,
            update_tracking=False,
        )
        self.conditions.slewtime = slewtimes

        # Let's get the sun and moon
        sun_moon_info = self.almanac.get_sun_moon_positions(self.mjd)
        # convert these to scalars
        for key in sun_moon_info:
            sun_moon_info[key] = sun_moon_info[key].max()
        self.conditions.moon_phase = sun_moon_info["moon_phase"]

        self.conditions.moon_alt = sun_moon_info["moon_alt"]
        self.conditions.moon_az = sun_moon_info["moon_az"]
        self.conditions.moon_ra = sun_moon_info["moon_RA"]
        self.conditions.moon_dec = sun_moon_info["moon_dec"]
        self.conditions.sun_alt = sun_moon_info["sun_alt"]
        self.conditions.sun_az = sun_moon_info["sun_az"]
        self.conditions.sun_ra = sun_moon_info["sun_RA"]
        self.conditions.sun_dec = sun_moon_info["sun_dec"]

        self.conditions.lmst = calc_lmst(self.mjd, self.site.longitude_rad)

        self.conditions.tel_ra = self.observatory.current_ra_rad
        self.conditions.tel_dec = self.observatory.current_dec_rad
        self.conditions.tel_alt = self.observatory.last_alt_rad
        self.conditions.tel_az = self.observatory.last_az_rad

        self.conditions.rot_tel_pos = self.observatory.last_rot_tel_pos_rad
        self.conditions.cumulative_azimuth_rad = self.observatory.cumulative_azimuth_rad

        # Add in the almanac information
        self.conditions.sunset = self.almanac.sunsets["sunset"][self.almanac_indx]
        self.conditions.sun_n12_setting = self.almanac.sunsets["sun_n12_setting"][self.almanac_indx]
        self.conditions.sun_n18_setting = self.almanac.sunsets["sun_n18_setting"][self.almanac_indx]
        self.conditions.sun_n18_rising = self.almanac.sunsets["sun_n18_rising"][self.almanac_indx]
        self.conditions.sun_n12_rising = self.almanac.sunsets["sun_n12_rising"][self.almanac_indx]
        self.conditions.sunrise = self.almanac.sunsets["sunrise"][self.almanac_indx]
        self.conditions.moonrise = self.almanac.sunsets["moonrise"][self.almanac_indx]
        self.conditions.moonset = self.almanac.sunsets["moonset"][self.almanac_indx]
        sun_moon_info_start_of_night = self.almanac.get_sun_moon_positions(self.conditions.sunset)
        self.conditions.moon_phase_sunset = sun_moon_info_start_of_night["moon_phase"]

        # Telescope limits
        self.conditions.sky_az_limits = self.sky_az_limits
        self.conditions.sky_alt_limits = self.sky_alt_limits
        self.conditions.tel_alt_limits = self.tel_alt_limits
        self.conditions.tel_az_limits = self.tel_az_limits

        # Planet positions from almanac
        self.conditions.planet_positions = self.almanac.get_planet_positions(self.mjd)

        # See if there are any ToOs to include
        if self.sim__to_o is not None:
            toos = self.sim__to_o(self.mjd)
            if toos is not None:
                self.conditions.targets_of_opportunity = toos

        if self.wind_data is not None:
            wind_speed, wind_direction = self.wind_data(current_time)
            self.conditions.wind_speed = wind_speed
            self.conditions.wind_direction = wind_direction

        if self.cloud_maps is not None:
            self.conditions.cloud_maps = self.cloud_maps

        return self.conditions

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        self.almanac_indx = self.almanac.mjd_indx(value)
        self.night = np.floor(self.mjd - self.mjd_start).astype(int)

    def observation_add_data(self, observation):
        """
        Fill in the metadata for a completed observation
        """
        current_time = Time(self.mjd, format="mjd")

        observation["clouds"] = self.cloud_data(current_time)
        observation["airmass"] = 1.0 / np.cos(np.pi / 2.0 - observation["alt"])
        # Seeing
        fwhm_500 = self.seeing_data(current_time)
        seeing_dict = self.seeing_model(fwhm_500, observation["airmass"])
        observation["FWHMeff"] = seeing_dict["fwhmEff"][self.seeing_indx_dict[observation["band"][0]]]
        observation["FWHM_geometric"] = seeing_dict["fwhmGeom"][self.seeing_indx_dict[observation["band"][0]]]
        observation["FWHM_500"] = fwhm_500

        observation["night"] = self.night
        observation["mjd"] = self.mjd

        if self.sky_model is not None:
            hpid = _ra_dec2_hpid(self.sky_model.nside, observation["RA"], observation["dec"])
            observation["skybrightness"] = self.sky_model.return_mags(
                self.mjd, indx=[hpid], extrapolate=True
            )[observation["band"][0]]

        observation["fivesigmadepth"] = m5_flat_sed(
            observation["band"][0],
            observation["skybrightness"],
            observation["FWHMeff"],
            observation["exptime"] / observation["nexp"],
            observation["airmass"],
            nexp=observation["nexp"],
        )
        # If there is cloud extinction, apply it.
        if self.cloud_maps is not None:
            hpid = _ra_dec2_hpid(self.sky_model.nside, observation["RA"], observation["dec"])
            cloud_extinction = self.cloud_maps.extinction_closest(self.mjd, hpid)
            observation["fivesigmadepth"] -= cloud_extinction
            observation["cloud_extinction"] = cloud_extinction

        lmst = calc_lmst(self.mjd, self.site.longitude_rad)
        observation["lmst"] = lmst

        sun_moon_info = self.almanac.get_sun_moon_positions(self.mjd)
        observation["sunAlt"] = sun_moon_info["sun_alt"]
        observation["sunAz"] = sun_moon_info["sun_az"]
        observation["sunRA"] = sun_moon_info["sun_RA"]
        observation["sunDec"] = sun_moon_info["sun_dec"]
        observation["moonAlt"] = sun_moon_info["moon_alt"]
        observation["moonAz"] = sun_moon_info["moon_az"]
        observation["moonRA"] = sun_moon_info["moon_RA"]
        observation["moonDec"] = sun_moon_info["moon_dec"]
        observation["moonDist"] = _angular_separation(
            observation["RA"],
            observation["dec"],
            observation["moonRA"],
            observation["moonDec"],
        )
        observation["solarElong"] = _angular_separation(
            observation["RA"],
            observation["dec"],
            observation["sunRA"],
            observation["sunDec"],
        )
        observation["moonPhase"] = sun_moon_info["moon_phase"]

        observation["ID"] = self.obs_id_counter
        self.obs_id_counter += 1

        return observation

    def check_up(self, mjd):
        """See if we are in downtime

        Returns
        --------
        is_up, mjd : `bool`, `float`
            Returns (True, current_mjd) if telescope is up
            and (False, downtime_ends_mjd) if in downtime
        """

        result = True, mjd
        indx = np.searchsorted(self.downtimes["start"], mjd, side="right") - 1
        if indx >= 0:
            if mjd < self.downtimes["end"][indx]:
                result = False, self.downtimes["end"][indx]
        return result

    def check_mjd(self, mjd, cloud_skip=20.0):
        """See if an mjd is ok to observe

        Parameters
        ----------
        cloud_skip : float (20)
            How much time to skip ahead if it's cloudy (minutes)

        Returns
        -------
        mjd_ok : `bool`
        mdj : `float`
            If True, the input mjd. If false, a good mjd to skip
            forward to.
        """
        passed = True
        new_mjd = mjd + 0

        # Maybe set this to a while loop to make sure we don't
        # land on another cloudy time?
        # or just make this an entire recursive call?
        clouds = self.cloud_data(Time(mjd, format="mjd"))

        if clouds > self.cloud_limit:
            passed = False
            while clouds > self.cloud_limit:
                new_mjd = new_mjd + cloud_skip / 60.0 / 24.0
                clouds = self.cloud_data(Time(new_mjd, format="mjd"))
        alm_indx = np.searchsorted(self.almanac.sunsets[self.starting_time_key], mjd, side="right") - 1
        # at the end of the night, advance to the next setting twilight
        if mjd > self.almanac.sunsets[self.ending_time_key][alm_indx]:
            passed = False
            new_mjd = self.almanac.sunsets[self.starting_time_key][alm_indx + 1]
        if mjd < self.almanac.sunsets[self.starting_time_key][alm_indx]:
            passed = False
            new_mjd = self.almanac.sunsets[self.starting_time_key][alm_indx + 1]
        # We're in a down time, if down, advance to the end of the downtime
        if not self.check_up(mjd)[0]:
            passed = False
            new_mjd = self.check_up(mjd)[1]
        # recursive call to make sure we skip far enough ahead
        if not passed:
            while not passed:
                passed, new_mjd = self.check_mjd(new_mjd)
            return False, new_mjd
        else:
            return True, mjd

    def _update_rot_sky_pos(self, observation, try_shifts=np.array([-np.pi / 2, np.pi / 2, -np.pi, np.pi])):
        """Update the rotSkyPos value to make sure it is observable.

        Parameters
        ----------
        observation : `rubin_scheduler.scheduler.ObservationArray`
            The observation to fix rotSkyPos on.
        try_shifts : `np.array`
            If the rotSkyPos value results in an invalid rotTelPos,
            rotate by the values in try_shifts and select the
            one that results in the smallest rotator slew.
            Default [-np.pi/2, np.pi/2, -np.pi, np.pi] (radians).
        """

        # Grab the rotator limit from the observatory model
        # Set so default rot_limit = [-90, 90]
        rot_limit = [
            self.observatory.telrot_minpos_rad,
            self.observatory.telrot_maxpos_rad,
        ]

        alt, az = _approx_ra_dec2_alt_az(
            observation["RA"],
            observation["dec"],
            self.site.latitude_rad,
            self.site.longitude_rad,
            self.mjd,
        )

        obs_pa = _approx_altaz2pa(alt, az, self.site.latitude_rad)

        if self.resolve_rotskypos:
            if not np.isfinite(observation["rotSkyPos"]):
                warnings.warn("No finite rotSkyPos value, using rotSkyPos_desired")
                observation["rotSkyPos"] = observation["rotSkyPos_desired"]
            rottelpos = self.rc._rotskypos2rottelpos(observation["rotSkyPos"], obs_pa)

            if (rottelpos < rot_limit[0]) | (rottelpos > rot_limit[1]):
                potential_rottelpos = rottelpos + try_shifts
                valid = np.where((potential_rottelpos > rot_limit[0]) & (potential_rottelpos < rot_limit[1]))[
                    0
                ]
                potential_rottelpos = potential_rottelpos[valid]
                # Which potential rot_tel_pos is closest to the
                # current rot_tel_pos?
                if self.observatory.current_rot_sky_pos_rad is None:
                    current_rot_tel_pos = 0
                else:
                    current_rot_tel_pos = self.rc._rotskypos2rottelpos(
                        self.observatory.current_rot_sky_pos_rad, obs_pa
                    )
                ang_diff = np.abs(smallest_signed_angle(current_rot_tel_pos, potential_rottelpos))
                indx = np.where(ang_diff == np.min(ang_diff))[0]
                rottelpos = potential_rottelpos[indx]

            observation["rotSkyPos"] = self.rc._rottelpos2rotskypos(rottelpos, obs_pa)

        else:
            # If the observation has a rotTelPos set,
            # use it to compute rotSkyPos
            if np.isfinite(observation["rotTelPos"]):
                observation["rotSkyPos"] = self.rc._rottelpos2rotskypos(observation["rotTelPos"], obs_pa)
                observation["rotTelPos"] = np.nan
            else:
                # Try to fall back to rotSkyPos_desired
                possible_rot_tel_pos = self.rc._rotskypos2rottelpos(observation["rotSkyPos_desired"], obs_pa)
                # If in range, use rotSkyPos_desired for rotSkyPos
                if (possible_rot_tel_pos > rot_limit[0]) & (possible_rot_tel_pos < rot_limit[1]):
                    observation["rotSkyPos"] = observation["rotSkyPos_desired"]
                    observation["rotTelPos"] = np.nan
                else:
                    # Fall back to the backup rotation angle if needed.
                    observation["rotSkyPos"] = np.nan
                    observation["rotTelPos"] = observation["rotTelPos_backup"]

        return observation

    def observe(self, observation):
        """Try to make an observation

        Returns
        -------
        observation : observation object
            None if there was no observation taken. Completed
            observation with meta data filled in.
        new_night : bool
            Have we started a new night.
        """

        start_night = self.night.copy()

        observation = self._update_rot_sky_pos(observation)

        # If there has been a long gap, assume telescope stopped
        # tracking and parked
        gap = self.mjd - self.observatory.last_mjd
        if gap > self.park_after:
            self.observatory.park()

        # Compute what alt,az we have tracked to (or are parked at)
        start_alt, start_az, start_rot_tel_pos = self.observatory.current_alt_az(self.mjd)
        # Slew to new position and execute observation. Use the
        # requested rotTelPos position, obsevation['rotSkyPos'] will
        # be ignored.
        slewtime, visittime = self.observatory.observe(
            observation,
            self.mjd,
            rot_tel_pos=observation["rotTelPos"],
            lax_dome=self.lax_dome,
        )

        # inf slewtime means the observation failed (probably outside
        # alt limits)
        if ~np.all(np.isfinite(slewtime)):
            return None, False

        observation_worked, new_mjd = self.check_mjd(self.mjd + (slewtime + visittime) / 24.0 / 3600.0)

        if observation_worked:
            observation["visittime"] = visittime
            observation["slewtime"] = slewtime
            observation["slewdist"] = _angular_separation(
                start_az,
                start_alt,
                self.observatory.last_az_rad,
                self.observatory.last_alt_rad,
            )
            self.mjd = self.mjd + slewtime / 24.0 / 3600.0
            # Reach into the observatory model to pull out the
            # relevant data it has calculated
            # Not bothering to fetch alt,az,pa,rottelpos as those
            # were computed before the slew was executed
            # so will be off by seconds to minutes. And they
            # shouldn't be needed by the scheduler.

            observation["rotSkyPos"] = self.observatory.current_rot_sky_pos_rad
            observation["cummTelAz"] = self.observatory.cumulative_azimuth_rad

            # But we do need the altitude to
            # get the airmass and then 5-sigma depth
            # this altitude should get clobbered later
            # by sim_runner.
            observation["alt"] = self.observatory.last_alt_rad

            # Metadata on observation is after slew and settle,
            # so at start of exposure.
            result = self.observation_add_data(observation)
            self.mjd = self.mjd + visittime / 24.0 / 3600.0
            new_night = False
        else:
            result = None
            self.observatory.park()
            # Skip to next legitimate mjd
            self.mjd = new_mjd
            now_night = self.night
            if now_night == start_night:
                new_night = False
            else:
                new_night = True

        return result, new_night

    # methods to reach through and adjust the kinematic model if desired
    def setup_camera(self, **kwargs):
        self.observatory.setup_camera(**kwargs)

    def setup_dome(self, **kwargs):
        self.observatory.setup_dome(**kwargs)

    def setup_telescope(self, **kwargs):
        self.observatory.setup_telescope(**kwargs)

    def setup_setup_optics(self, **kwargs):
        self.observatory.setup_optics(**kwargs)
