import warnings

import numpy as np

from rubin_scheduler.scheduler.utils import smallest_signed_angle
from rubin_scheduler.utils import (
    Site,
    _approx_alt_az2_ra_dec,
    _approx_altaz2pa,
    _approx_ra_dec2_alt_az,
    rotation_converter,
)

from .jerk import jerk_time

__all__ = (
    "tma_movement",
    "rotator_movement",
    "KinemModel",
)
two_pi = 2.0 * np.pi


def tma_movement(percent=70):
    """Get a dictionary of parameters to pass to `setup_telescope`
     defining altitude and azimuth speed, acceleration, and jerk
     in terms of 'percent' of total performance.

     Parameters
     ----------
     percent : `float`, optional
        Default performance for the scheduler simulations for operations
        has been 70% (70, default).
        Expected performance at the start of comcam on-sky
        science operations is about 10%.

    Returns
    -------
    tma : `dict` {`str`: `float`}
        A dictionary which can be passed as kwargs to
        KinematicModel.setup_telescope(**tma).
    """
    # See https://confluence.lsstcorp.org/display/LSSTCOM/TMA+Motion+Settings
    # Expected performance at end of comcam on-sky is probably 10%
    if percent > 125:
        percent = 125
        print("Cannot exceed 125 percent, by requirements.")
    tma = {}
    scale = percent / 100.0
    tma["azimuth_maxspeed"] = np.min([10.0 * scale, 7.0])
    tma["azimuth_accel"] = 10.0 * scale
    tma["azimuth_jerk"] = np.max([1.0, 40.0 * scale])
    tma["altitude_maxspeed"] = 5.0 * scale
    tma["altitude_accel"] = 5.0 * scale
    tma["altitude_jerk"] = np.max([1.0, 20.0 * scale])
    tma["settle_time"] = 3.0
    return tma


def rotator_movement(percent=100):
    """Get a dictionary of parameters to pass to `setup_camera`
     defining rotator max speed, acceleration and jerk,
     in terms of 'percent' of total performance.

     Parameters
     ----------
     percent : `float`, optional
        Default performance for the scheduler simulations for operations
        has been 100% (100, default).
        Expected performance at the start of comcam on-sky
        science operations is approximately full performance.

    Returns
    -------
    rot : `dict` {`str`: `float`}
        A dictionary which can be passed as kwargs to
        KinematicModel.setup_camera(**rot).
    """
    # Kevin and Brian say these can run 100%
    # and are independent of TMA movement
    if percent > 125:
        percent = 125
        print("Cannot exceed 125 percent, by requirements.")
    rot = {}
    rot["maxspeed"] = 3.5 * percent / 100
    rot["accel"] = 1.0 * percent / 100
    rot["jerk"] = 4.0 * percent / 100
    return rot


class Radec2altazpa:
    """Class to make it easy to swap in different alt/az conversion if
    wanted."""

    def __init__(self, location):
        self.location = location

    def __call__(self, ra, dec, mjd):
        alt, az, pa = _approx_ra_dec2_alt_az(
            ra, dec, self.location.lat_rad, self.location.lon_rad, mjd, return_pa=True
        )
        return alt, az, pa


class KinemModel:
    """A Kinematic model of the telescope.

    Parameters
    ----------
    location : `astropy.coordinates.EarthLocation`
        The location of the telescope.
        If None, defaults to rubin_scheduler.utils.Site info
    park_alt : `float` (86.5)
        The altitude the telescope gets parked at (degrees)
    park_az : `float` (0)
        The azimuth for telescope park position (degrees)
    start_band : `str` ('r')
        The band that gets loaded when the telescope is parked
    mjd0 : `float` (0)
        The MJD to assume we are starting from
    telescope : `str'
        The telescope name to use for sky rotation conversion. Default "rubin"

    Note
    ----
    Note there are additional parameters in the methods setup_camera,
    setup_dome, setup_telescope, and setup_optics.
    """

    def __init__(self, location=None, park_alt=86.5, park_az=0.0, start_band="r", mjd0=0, telescope="rubin"):
        self.park_alt_rad = np.radians(park_alt)
        self.park_az_rad = np.radians(park_az)
        self.start_band = start_band
        self.current_band = self.start_band
        if location is None:
            self.location = Site("LSST")
            self.location.lat_rad = np.radians(self.location.latitude)
            self.location.lon_rad = np.radians(self.location.longitude)
        # Our RA,Dec to Alt,Az converter
        self.radec2altaz = Radec2altazpa(self.location)

        self.setup_camera()
        self.setup_dome()
        self.setup_telescope()
        self.setup_optics()

        # Park the telescope
        self.park()
        self.last_mjd = mjd0

        # Rotation conversion
        self.rc = rotation_converter(telescope=telescope)

    def mount_bands(self, band_list):
        """Change which bands are mounted

        Parameters
        ----------
        band_list : `list` [`str`]
            List of the mounted bands.
        """
        self.mounted_bands = band_list
        # Make sure we're using one of the available bands.
        if self.current_band not in self.mounted_bands:
            self.current_band = self.mounted_bands[-1]
        if self.start_band not in self.mounted_bands:
            self.start_band = self.mounted_bands[-1]

    def setup_camera(
        self,
        readtime=3.07,
        shuttertime=1.0,
        band_changetime=120.0,
        fov=3.5,
        rotator_min=-90,
        rotator_max=90,
        maxspeed=3.5,
        accel=1.0,
        shutter_2motion_min_time=15.0,
        jerk=4.0,
    ):
        """Configure the camera.

        Parameters
        ----------
        readtime : `float`
            The readout time of the CCDs (seconds)
        shuttertime : `float`
            The time it takes the shutter to go from closed to fully open
            (seconds)
        band_changetime : `float`
            The time it takes to change bands (seconds)
        fov : `float`
            The camera field of view (degrees)
        rotator_min : `float`
            The minimum angle the camera rotator (rotTelPos) can move to
            (degrees)
        rotator_max : `float`
            The maximum angle the camera rotator (rotTelPos) can move to
            degrees)
        maxspeed : `float`
            The maximum speed of the rotator (degrees/s)
        accel : `float`
            The acceleration of the rotator (degrees/s^2)
        jerk : `float`
            The jerk of the rotator (degrees/s^3). None treats
            jerk as infinite. Default 4.0.
        shutter_2motion_min_time : `float`
            The time required for two shutter motions (seconds). If one
            takes a 1-snap 10s exposure, there will be a 5s of overhead
            before the next exposure can start.
        """
        self.readtime = readtime
        self.shuttertime = shuttertime
        self.band_changetime = band_changetime
        self.camera_fov = np.radians(fov)

        self.telrot_minpos_rad = np.radians(rotator_min)
        self.telrot_maxpos_rad = np.radians(rotator_max)
        self.telrot_maxspeed_rad = np.radians(maxspeed)
        self.telrot_accel_rad = np.radians(accel)
        self.telrot_jerk_rad = np.radians(jerk) if jerk is not None else None
        self.shutter_2motion_min_time = shutter_2motion_min_time
        self.mounted_bands = ["u", "g", "r", "i", "y"]

    def setup_dome(
        self,
        altitude_maxspeed=1.75,
        altitude_accel=0.875,
        altitude_jerk=None,
        altitude_freerange=0.0,
        azimuth_maxspeed=1.5,
        azimuth_accel=0.75,
        azimuth_jerk=None,
        azimuth_freerange=4.0,
        settle_time=1.0,
    ):
        """Configure the dome.

        Parameters
        ----------
        altitude_maxspeed : `float`
            Maximum speed for altitude movement (degrees/second)
        altitude_accel : `float`
            Maximum acceleration for altitude movement (degrees/second**2)
        altitude_jerk : `float`
            The jerk for the altitude movement (degrees/second**3).
            None treats jerk as infinite. Default None
        altitude_freerange : `float`
            The range over which there is 0 delay
        azimuth_maxspeed : `float`
            Maximum speed for azimuth movement (degrees/second)
        azimuth_accel : `float`
            Maximum acceleration for azimuth movement (degrees/second**2)
        azimuth_jerk : `float`
            The jerk of the azimuth movement (degrees/second**3). Default
            of None treats jerk as infinite.
        azimuth_freerange : `float`
            The range in which there is 0 delay
        settle_time : `float`
            Settle time after movement (seconds)
        """
        self.domalt_maxspeed_rad = np.radians(altitude_maxspeed)
        self.domalt_accel_rad = np.radians(altitude_accel)
        self.domalt_jerk_rad = np.radians(altitude_jerk) if altitude_jerk is not None else None
        self.domalt_free_range = np.radians(altitude_freerange)
        self.domaz_maxspeed_rad = np.radians(azimuth_maxspeed)
        self.domaz_accel_rad = np.radians(azimuth_accel)
        self.domaz_jerk_rad = np.radians(azimuth_jerk) if azimuth_jerk is not None else None
        self.domaz_free_range = np.radians(azimuth_freerange)
        self.domaz_settletime = settle_time

    def setup_telescope(
        self,
        altitude_minpos=20.0,
        altitude_maxpos=86.5,
        azimuth_minpos=-250.0,
        azimuth_maxpos=250.0,
        altitude_maxspeed=3.5,
        altitude_accel=3.5,
        altitude_jerk=14.0,
        azimuth_maxspeed=7.0,
        azimuth_accel=7.0,
        azimuth_jerk=28.0,
        settle_time=3.0,
    ):
        """Configure the telescope (TMA) movement and position.

        Parameters
        ----------
        altitude_minpos : `float`
            Minimum altitude for the telescope (degrees).
        altitude_maxpos : `float`
            Maximum altitude for the telescope (degrees).
            Maximum position of 86.5 is limited due to slew
            requirements near the zenith with an alt-az mount.
        azimuth_minpos : `float`
            Minimum azimuth position (degrees).
            Note this value is related to cumulative azimuth range,
            for cable wrap.
        azimuth_maxpos : `float`
            Maximum azimuth position (degrees).
            Note this value is related to cumulative azimuth range,
            for cable wrap. Defaults -250/250 include 0-360
            for all-sky azimuth positions, reachable via multiple
            routes.
        altitude_maxspeed : `float`
            Maximum speed for altitude movement (degrees/second)
        altitude_accel : `float`
            Maximum acceleration for altitude movement (degrees/second**2)
        altitude_jerk : `float`
            Jerk for altitude movement (degrees/second**2). None will
            treat jerk as infinite. Default 20.
        azimuth_maxspeed : `float`
            Maximum speed for azimuth movement (degrees/second)
        azimuth_accel : `float`
            Maximum acceleration for azimuth movement (degrees/second**2)
        azimuth_jerk : `float`
            Jerk for azimuth movement (degrees/second**2). None will
            treat jerk as infinite. Default 40.
        settle_time : `float`
            Settle time required for telescope after movement (seconds)
            See also `tma_movement` for definitions for the speed,
            acceleration and jerk.
        """
        self.telalt_minpos_rad = np.radians(altitude_minpos)
        self.telalt_maxpos_rad = np.radians(altitude_maxpos)
        self.telaz_minpos_rad = np.radians(azimuth_minpos)
        self.telaz_maxpos_rad = np.radians(azimuth_maxpos)
        self.telalt_maxspeed_rad = np.radians(altitude_maxspeed)
        self.telalt_accel_rad = np.radians(altitude_accel)
        self.telalt_jerk_rad = np.radians(altitude_jerk) if altitude_jerk is not None else None
        self.telaz_maxspeed_rad = np.radians(azimuth_maxspeed)
        self.telaz_accel_rad = np.radians(azimuth_accel)
        self.telaz_jerk_rad = np.radians(azimuth_jerk) if azimuth_jerk is not None else None
        self.mount_settletime = settle_time

    def setup_optics(self, ol_slope=1.0 / 3.5, cl_delay=[0.0, 36.0], cl_altlimit=[0.0, 9.0, 90.0]):
        """Configure parameters for the optics closed and open loops.

        Parameters
        ----------
        ol_slope : `float` (1.0/3.5)
            seconds/degree in altitude slew.
        cl_delay : list ([0.0, 36])
            The delays for closed optics loops (seconds)
        cl_altlimit : list ([0.0, 9.0, 90.0])
            The altitude limits (degrees) for performing closed optice loops.
            Should be one element longer than cl_delay.

        Note
        ----
        A given movement in altitude will cover X degrees;
        if X > cl_altlimit[i] there is
        an additional delay of cl_delay[i]
        """
        # ah, 1./np.radians(1)=np.pi/180
        self.optics_ol_slope = ol_slope / np.radians(1.0)
        self.optics_cl_delay = cl_delay
        self.optics_cl_altlimit = np.radians(cl_altlimit)

    def park(self):
        """Put the telescope in the park position."""
        # I'm going to ignore that the old model had the dome altitude at 90
        # and telescope altitude 86 for park.
        # We should usually be dome az limited anyway, so this should be a
        # negligible approximation.
        self.parked = True

        # We have no current position we are tracking
        self.current_ra_rad = None
        self.current_dec_rad = None
        self.current_rot_sky_pos_rad = None
        self.cumulative_azimuth_rad = 0

        # The last position we were at (or the current if we are parked)
        self.last_az_rad = self.park_az_rad
        self.last_alt_rad = self.park_alt_rad
        self.last_rot_tel_pos_rad = 0

        # Any overhead that must happen before next exposure can start. Slew
        # motions are allowed during the overhead time
        self.overhead = 0.0

        # Don't leave random band in overnight
        self.current_band = self.start_band

    def current_alt_az(self, mjd):
        """return the current alt az position that we have tracked to."""
        if self.parked:
            return self.last_alt_rad, self.last_az_rad, self.last_rot_tel_pos_rad
        else:
            alt_rad, az_rad, pa = self.radec2altaz(self.current_ra_rad, self.current_dec_rad, mjd)
            rot_tel_pos = self.rc._rotskypos2rottelpos(self.current_rot_sky_pos_rad, pa)
            return alt_rad, az_rad, rot_tel_pos

    def slew_times(
        self,
        ra_rad,
        dec_rad,
        mjd,
        rot_sky_pos=None,
        rot_tel_pos=None,
        bandname=np.array(["r"]),
        lax_dome=True,
        slew_while_changing_filter=False,
        constant_band_changetime=True,
        alt_rad=None,
        az_rad=None,
        starting_alt_rad=None,
        starting_az_rad=None,
        starting_rot_tel_pos_rad=None,
        update_tracking=False,
    ):
        """Calculates slew time to a series of alt/az/band positions
        from the current position (stored internally).

        Assumptions (currently):
        Assumes we have been tracking on ra,dec,rot_sky_pos position.
        Ignores the motion of the sky while we are slewing
        (this approx should probably average out over time).
        No checks for if we have tracked beyond limits.
        (this assumes folks put telescope in park if there's a long gap.)
        Assumes the camera rotator never needs to (or can't) do a slew
        over 180 degrees.

        Calculates the slew time necessary to get from current state
        to alt2/az2/band2. The readout time is NOT included in the slewtime,
        as you could slew without taking an exposure. However, the readout
        time (from the previous exposure) will be included in `self.overhead`
        when calculating visit+slew using `observe`, and `self.overhead`
        will be a minimum value for the slewtime.

        Parameters
        ----------
        ra_rad : `np.ndarray`
            The RA(s) of the location(s) we wish to slew to (radians)
        dec_rad : `np.ndarray`
            The declination(s) of the location(s) we wish to slew to (radians)
        mjd : `float`
            The current modified julian date (days)
        rot_sky_pos : `np.ndarray`
            The desired rot_sky_pos(s) (radians).
            Angle between up on the chip and North.
            Note, it is possible to set a rot_sky_pos outside the allowed
            camera rotator range, in which case the slewtime will be masked.
            If both rot_sky_pos and rot_tel_pos are set,
            rot_tel_pos will be used.
        rot_tel_pos : `np.ndarray`
            The desired rot_tel_pos(s) (radians).
            This overrides rot_sky_pos, if set.
            If neither rot_sky_pos nor rot_tel_pos are set, no rotation
            time is added (equivalent to using current rot_tel_pos).
        bandname : `str` or None
            The band(s) of the desired observations.
            Set to None to compute only telescope and dome motion times.
        alt_rad : `np.ndarray`
            The altitude(s) of the destination pointing(s) (radians).
            Will override ra_rad,dec_rad if provided.
        az_rad : `np.ndarray`
            The azimuth(s) of the destination pointing(s) (radians).
            Will override ra_rad,dec_rad if provided.
        lax_dome : `bool`, default True
            If True, allow the dome to creep, model a dome slit, and don't
            require the dome to settle in azimuth. If False, adhere to the
            way SOCS calculates slew times (as of June 21 2017) and do not
            allow dome creep.
        slew_while_changing_filter : `bool`, default False
            If False, slew first and then stop to change the filter.
            If True, change filter in parallel with slewing.
        constant_band_changetime : `bool`, default True
            If True, use a constant (average) value for the band change time.
            The value will be self.band_changetime.
            If False, calculate rotator + approximate carousel swap time.
        starting_alt_rad : `float` (None)
            The starting altitude for the slew (radians).
            If None, will use internally stored last pointing.
        starting_az_rad : `float` (None)
            The starting azimuth for the slew (radians).
            If None, will use internally stored last pointing.
        starting_rot_tel_pos_rad : `float` (None)
            The starting camera rotation for the slew (radians).
            If None, will use internally stored last pointing.
        update_tracking : `bool` (False)
            If True, update the internal attributes to say we are tracking
            the specified RA,Dec,RotSkyPos position.


        Returns
        -------
        slew_time : `np.ndarray`
            The number of seconds between the two specified exposures.
            Will be np.nan or np.inf if slew is not possible.


        Notes
        -----
        When using the KinematicModel with the ModelObservatory,
        actual slews + observations are calculated using
        `KinematicModel.observe`. This adds any additional time due to
        shutter motion limits or readout into `self.overhead` which
        is then applied as a minimum to the slew to the next pointing,
        appropriately attributing the slewtime (slew is always counted
        as the time *to* a pointing).
        When using the KinematicModel to just calculate slewtime
        between real visits, either use `observe` or use `slewtime`
        directly but add the readout time to `KinematicModel.overhead`
        for each visit.
        """
        if bandname is None:
            bandname = self.current_band
        elif bandname not in self.mounted_bands:
            return np.nan

        # if alt,az not provided, then calculate from RA,Dec
        if alt_rad is None:
            alt_rad, az_rad, pa = self.radec2altaz(ra_rad, dec_rad, mjd)
        else:
            pa = _approx_altaz2pa(alt_rad, az_rad, self.location.lat_rad)
            if update_tracking:
                ra_rad, dec_rad = _approx_alt_az2_ra_dec(
                    alt_rad, az_rad, self.location.lat_rad, self.location.lon_rad, mjd
                )

        # If either rot_tel_pos or rot_sky_pos are set, we will
        # calculate slewtime with rotator movement.
        # Setting rot_tel_pos allows any slew position on-sky.
        # Setting rot_sky_pos can restrict slew range on-sky
        # as some rot_tel_pos values that result will be out of bounds.
        # Use rot_tel_pos first if available, to override rot_sky_pos.
        if (rot_tel_pos is not None) | (rot_sky_pos is not None):
            if rot_tel_pos is not None and np.isfinite(rot_tel_pos):
                rot_sky_pos = self.rc._rottelpos2rotskypos(rot_tel_pos, pa)
            else:
                rot_tel_pos = self.rc._rotskypos2rottelpos(rot_sky_pos, pa)

        # Find the current location of the telescope.
        if starting_alt_rad is None:
            if self.parked:
                starting_alt_rad = self.park_alt_rad
                starting_az_rad = self.park_az_rad
            else:
                starting_alt_rad, starting_az_rad, starting_pa = self.radec2altaz(
                    self.current_ra_rad, self.current_dec_rad, mjd
                )

        # Now calculate how far we need to move,
        # in altitude, azimuth, and camera rotator (if applicable).
        # Delta Altitude
        delta_alt = np.abs(alt_rad - starting_alt_rad)
        # Delta Azimuth - there are two different directions
        # possible to travel for azimuth. First calculate the shortest.
        delta_az_short = smallest_signed_angle(starting_az_rad, az_rad)
        # And then calculate the longer
        delta_az_long = np.where(delta_az_short < 0, two_pi + delta_az_short, delta_az_short - two_pi)
        # Slew can go long or short direction, but azimuth range
        # could limit which is possible.
        # e.g. 70 degrees reached by going the long way around from 0 means
        # the cumulative azimuth is 290, but if we went the short way it would
        # be 70 .. absolute azimuth is still 70. Direction of previous
        # slews is also important.
        # First evaluate if available azimuth range is > 360 degrees --
        if np.abs(self.telaz_maxpos_rad - self.telaz_minpos_rad) >= two_pi:
            # Can spin past 360 degrees, track cumulative azimuth
            # Note that minpos will be less than maxpos always in this case.
            cummulative_az_short = delta_az_short + self.cumulative_azimuth_rad
            out_of_bounds = np.where(
                (cummulative_az_short < self.telaz_minpos_rad)
                | (cummulative_az_short > self.telaz_maxpos_rad)
            )[0]
            # Set short direction out of bounds azimuths to infinite distance.
            delta_az_short[out_of_bounds] = np.inf
            cummulative_az_long = delta_az_long + self.cumulative_azimuth_rad
            out_of_bounds = np.where(
                (cummulative_az_long < self.telaz_minpos_rad) | (cummulative_az_long > self.telaz_maxpos_rad)
            )[0]
            # Set long direction out of bounds azimuths to infinite distance
            delta_az_long[out_of_bounds] = np.inf
            # Now pick the shortest allowable direction
            # (use absolute value of each az, because values can be negative)
            delta_aztel = np.where(
                np.abs(delta_az_short) < np.abs(delta_az_long),
                delta_az_short,
                delta_az_long,
            )
            az_flag = "delta"
        # Now evaluate the options if we have an impaired telescope with
        # azimuth range < 360 degrees.
        else:
            # Note that minpos will be the starting angle, but
            # depending on direction available - maxpos might be < minpos.
            az_range = (self.telaz_maxpos_rad - self.telaz_minpos_rad) % (two_pi)
            out_of_bounds = np.where((az_rad - self.telaz_minpos_rad) % (two_pi) > az_range)[0]
            d1 = (az_rad - self.telaz_minpos_rad) % (two_pi)
            d2 = (starting_az_rad - self.telaz_minpos_rad) % (two_pi)
            delta_aztel = d2 - d1
            delta_aztel[out_of_bounds] = np.inf
            az_flag = "restricted"

        # Calculate time to slew to this position.
        tel_alt_slew_time = jerk_time(
            delta_alt, self.telalt_maxspeed_rad, self.telalt_accel_rad, self.telalt_jerk_rad
        )
        tel_az_slew_time = jerk_time(
            np.abs(delta_aztel), self.telaz_maxspeed_rad, self.telaz_accel_rad, self.telaz_jerk_rad
        )
        tot_tel_time = np.maximum(tel_alt_slew_time, tel_az_slew_time)

        # Time for open loop optics correction
        ol_time = delta_alt / self.optics_ol_slope
        tot_tel_time += ol_time
        # Add time for telescope settle.
        # note, this means we're going to have a settle time even for very
        # small slews like dithering.
        settle_and_ol = np.where(tot_tel_time > 0)
        tot_tel_time[settle_and_ol] += np.maximum(0, self.mount_settletime - ol_time[settle_and_ol])

        # And any leftover overhead sets a minimum on the total telescope
        # time. The overhead includes the readout time from the previous
        # observation and any rate-limitation from the shutter motion.
        # self.overhead is set in the `observe` method by default.
        tot_tel_time = np.maximum(self.overhead, tot_tel_time)

        # now compute dome slew time
        # the dome can spin all the way around, so we will let it go the
        # shortest angle, even if the telescope has to unwind
        delta_az = np.abs(smallest_signed_angle(starting_az_rad, az_rad))
        if lax_dome:
            # model dome creep, dome slit, and no azimuth settle
            # if we can fit both exposures in the dome slit, do so
            same_dome = np.where(delta_alt**2 + delta_az**2 < self.camera_fov**2)

            # else, we take the minimum time from two options:
            # 1. assume we line up alt in the center of the dome slit so we
            #    minimize distance we have to travel in azimuth.
            # 2. line up az in the center of the slit
            # also assume:
            # * that we start out going maxspeed for both alt and az
            # * that we only just barely have to get the new field in the
            #   dome slit in one direction, but that we have to center the
            #   field in the other (which depends which of the two options
            #   used)
            # * that we don't have to slow down until after the shutter
            #   starts opening
            dom_delta_alt = delta_alt
            # on each side, we can start out with the dome shifted away from
            # the center of the field by an amount domSlitRadius - fovRadius
            dom_slit_diam = self.camera_fov / 2.0
            dom_delta_az = delta_az - 2 * (dom_slit_diam / 2 - self.camera_fov / 2)
            dom_alt_slew_time = dom_delta_alt / self.domalt_maxspeed_rad
            dom_az_slew_time = dom_delta_az / self.domaz_maxspeed_rad
            tot_dom_time1 = np.maximum(dom_alt_slew_time, dom_az_slew_time)

            dom_delta_alt = delta_alt - 2 * (dom_slit_diam / 2 - self.camera_fov / 2)
            dom_delta_az = delta_az
            dom_alt_slew_time = dom_delta_alt / self.domalt_maxspeed_rad
            dom_az_slew_time = dom_delta_az / self.domaz_maxspeed_rad
            tot_dom_time2 = np.maximum(dom_alt_slew_time, dom_az_slew_time)

            tot_dom_time = np.minimum(tot_dom_time1, tot_dom_time2)
            tot_dom_time[same_dome] = 0

        else:
            # the above models a dome slit and dome creep.
            # If this option is not available however:
            dom_alt_slew_time = jerk_time(
                delta_alt, self.domalt_maxspeed_rad, self.domalt_accel_rad, self.domalt_jerk_rad
            )
            dom_az_slew_time = jerk_time(
                delta_az, self.domaz_maxspeed_rad, self.domaz_accel_rad, self.domaz_jerk_rad
            )

            # Dome takes 1 second to settle in az
            dom_az_slew_time = np.where(
                dom_az_slew_time > 0,
                dom_az_slew_time + self.domaz_settletime,
                dom_az_slew_time,
            )
            tot_dom_time = np.maximum(dom_alt_slew_time, dom_az_slew_time)

        # Find the max of the above for slew time.
        slew_time = np.maximum(tot_tel_time, tot_dom_time)

        # If we want to include the camera rotation time
        # We will have already set rot_tel_pos above, if either
        # rot_sky_pos or rot_tel_pos is set.
        if rot_tel_pos is not None:
            outside_limits = np.where(
                (rot_tel_pos < self.telrot_minpos_rad) | (rot_tel_pos > self.telrot_maxpos_rad)
            )
            slew_time[outside_limits] = np.nan
            # If there was no kwarg for starting rotator position
            if starting_rot_tel_pos_rad is None:
                # If there is no current rot_sky_pos, we were parked
                if self.current_rot_sky_pos_rad is None:
                    current_rot_tel_pos = self.last_rot_tel_pos_rad
                else:
                    # We have been tracking, so rot_tel_pos needs to be updated
                    current_rot_tel_pos = self.rc._rotskypos2rottelpos(
                        self.current_rot_sky_pos_rad, starting_pa
                    )
            else:
                # kwarg overrides if it was supplied
                current_rot_tel_pos = starting_rot_tel_pos_rad
            delta_rotation = np.abs(smallest_signed_angle(current_rot_tel_pos, rot_tel_pos))
            rotator_time = jerk_time(
                delta_rotation, self.telrot_maxspeed_rad, self.telrot_accel_rad, self.telrot_jerk_rad
            )
            # Do not include rotator movement if changing band,
            # as rotator movement is baked into band change time.
            band_change = np.where(bandname != self.current_band)
            rotator_time[band_change] = 0
            slew_time = np.maximum(slew_time, rotator_time)

        # include band change time if necessary. Assume no band
        # change time needed if we are starting parked
        if not self.parked:
            band_change = np.where(bandname != self.current_band)
            if not constant_band_changetime and rot_tel_pos is not None:
                band_changetime = self.better_band_changetime(current_rot_tel_pos, rot_tel_pos)
            else:
                band_changetime = self.band_changetime
            if slew_while_changing_filter:
                slew_time[band_change] = np.maximum(slew_time[band_change], band_changetime)
            else:
                slew_time[band_change] = slew_time[band_change] + band_changetime

        # Add closed loop optics correction
        # Find the limit where we must add the delay
        cl_limit = self.optics_cl_altlimit[1]
        cl_delay = self.optics_cl_delay[1]
        close_loop = np.where(delta_alt >= cl_limit)
        slew_time[close_loop] += cl_delay

        # Mask min/max altitude limits so slewtime = np.nan
        outside_limits = np.where((alt_rad > self.telalt_maxpos_rad) | (alt_rad < self.telalt_minpos_rad))[0]
        slew_time[outside_limits] = np.nan
        # Azimuth limits already masked through cumulative azimuth
        # calculation above (it might be inf, not nan, so let's swap).
        slew_time = np.where(np.isfinite(slew_time), slew_time, np.nan)

        # Recreate how this happened to work previously with single targets
        if len(slew_time) == 1 and rot_tel_pos is not None:
            slew_time = slew_time[0]

        # Update the internal attributes to note that we are now pointing
        # and tracking at the requested RA,Dec,rot_sky_pos
        if update_tracking and np.isfinite(slew_time):
            self.current_ra_rad = ra_rad
            self.current_dec_rad = dec_rad
            self.current_rot_sky_pos_rad = rot_sky_pos
            self.parked = False
            # Handy to keep as reference, but not used for any calculations
            self.last_rot_tel_pos_rad = rot_tel_pos
            self.last_az_rad = az_rad
            self.last_alt_rad = alt_rad
            self.last_pa_rad = pa
            # Track the cumulative azimuth
            if az_flag == "restricted":
                self.cumulative_azimuth_rad = az_rad
            else:
                self.cumulative_azimuth_rad += delta_aztel
            self.current_band = bandname
            self.last_mjd = mjd

        return slew_time

    def better_band_changetime(self, rot_tel_prev, rot_tel_next, band_change=90):
        """A more precise estimate for the filter change time.

        The default of 120s is intended to capture an average rotator movement
        plus band change. In reality, the rotator moves from current position,
        to 0, changes the filter (either 60 or 90s), then moves to next
        rotation position.

        Parameters
        ----------
        rot_tel_prev : `float`
            Starting rotator position. Radians.
        rot_tel_next : `float`
            Ending rotator position. Radians.
        band_change : `float`
            The time to swap the filters in the carousel.
            This value depends on which slots are being changed, so
            could be 60 or 90s. Assume 90 for now. Seconds.

        Returns
        -------
        better_band_changetime : `float`
            An estimate of the filter change time required, in s.
        """
        # Rotate from previous angle to 0
        delta_rotation = np.abs(smallest_signed_angle(rot_tel_prev, np.array([0])))
        band_change_time = jerk_time(
            delta_rotation, self.telrot_maxspeed_rad, self.telrot_accel_rad, self.telrot_jerk_rad
        )
        # Add the carousel swap
        band_change_time += band_change
        # Rotate from 0 to new angle
        delta_rotation = np.abs(smallest_signed_angle(np.array([0]), rot_tel_next))
        band_change_time += jerk_time(
            delta_rotation, self.telrot_maxspeed_rad, self.telrot_accel_rad, self.telrot_jerk_rad
        )
        return band_change_time

    def visit_time(self, observation):
        # How long does it take to make an observation.
        # Includes shuttertime for each exposure and
        # all but last readout within same visit, as well as on-sky exptime
        visit_time = (
            observation["exptime"]
            + observation["nexp"] * self.shuttertime
            + max(observation["nexp"] - 1, 0) * self.readtime
        )
        return visit_time

    def shutter_stall(self, observation):
        """Time we need to stall after shutter closes to let things cool
        down."""
        result = 0.0
        delta_t = observation["exptime"] / observation["nexp"]
        if delta_t < self.shutter_2motion_min_time:
            result = self.shutter_2motion_min_time - delta_t
        return result

    def observe(self, observation, mjd, rot_tel_pos=None, lax_dome=True, slew_while_changing_filter=False):
        """Observe a target, and return the slewtime and visit time for
        the action

        If slew is not allowed, returns np.nan and does not update state.

        Calculates the visit time, taking into account the number of exposures
        and the readout time. It also imposes a minimum time between the
        start of exposures of `shutter_2motion_min_time`, adding a limit to
        the rate of acquiring observations.

        Parameters
        ----------
        observation : `ObservationArray`
            An observation array with the target to be observed.
        mjd : `float`
            The current MJD at the start of the slew.
        rot_tel_pos : `float` or None
            If specified, use rot_tel_pos for the rotator angle.
            Otherwise, will use rot_sky_pos from observation.
        lax_dome : `bool`, default True
            If True, allow the dome to creep, model a dome slit, and don't
            require the dome to settle in azimuth. If False, adhere to the
            way SOCS calculates slew times (as of June 21 2017) and do not
            allow dome creep.
        slew_while_changing_filter : `bool`, default False
            If False, slew first and then stop to change the filter.
            If True, change filter in parallel with slewing.
        """
        # Check on rate of shutter motion (but don't fail, just warn)
        if observation["nexp"] >= 2:
            shutter_rate = observation["exptime"] / observation["nexp"] + self.readtime * (
                observation["nexp"] - 1
            )
            if shutter_rate < self.shutter_2motion_min_time:
                msg = "%i exposures in %i seconds is violating number of shutter motion limit" % (
                    observation["nexp"],
                    observation["exptime"],
                )
                warnings.warn(msg)

        # Calculate slew time (without readout)
        slewtime = self.slew_times(
            observation["RA"],
            observation["dec"],
            mjd,
            rot_sky_pos=observation["rotSkyPos"],
            rot_tel_pos=rot_tel_pos,
            bandname=observation["band"],
            update_tracking=True,
            slew_while_changing_filter=slew_while_changing_filter,
            lax_dome=lax_dome,
        )

        # Calculate visit time (without last readout)
        visit_time = self.visit_time(observation)

        # Compute any overhead that is left over from this -
        # to be applied to next slewtime calculation.
        if ~np.isnan(slewtime):
            self.overhead = np.maximum(self.readtime, self.shutter_stall(observation))

        return slewtime, visit_time
