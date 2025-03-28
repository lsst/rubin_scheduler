__all__ = (
    "BaseBasisFunction",
    "HealpixLimitedBasisFunctionMixin",
    "ConstantBasisFunction",
    "SimpleArrayBasisFunction",
    "DelayStartBasisFunction",
    "AvoidFastRevisitsBasisFunction",
    "VisitRepeatBasisFunction",
    "M5DiffBasisFunction",
    "M5DiffAtHpixBasisFunction",
    "StrictBandBasisFunction",
    "StrictFilterBasisFunction",
    "BandChangeBasisFunction",
    "FilterChangeBasisFunction",
    "SlewtimeBasisFunction",
    "CadenceEnhanceBasisFunction",
    "CadenceEnhanceTrapezoidBasisFunction",
    "AzimuthBasisFunction",
    "AzModuloBasisFunction",
    "DecModuloBasisFunction",
    "MapModuloBasisFunction",
    "SeasonCoverageBasisFunction",
    "NObsPerYearBasisFunction",
    "CadenceInSeasonBasisFunction",
    "NearSunHighAirmassBasisFunction",
    "EclipticBasisFunction",
    "VisitGap",
    "NGoodSeeingBasisFunction",
    "AvoidDirectWind",
    "BalanceVisits",
    "RewardNObsSequence",
    "BandDistBasisFunction",
    "FilterDistBasisFunction",
    "RewardRisingBasisFunction",
    "send_unused_deprecation_warning",
)

import warnings

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from rubin_scheduler.scheduler import features
from rubin_scheduler.scheduler.utils import IntRounded, get_current_footprint
from rubin_scheduler.skybrightness_pre import dark_m5
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD, _hpid2_ra_dec


def send_unused_deprecation_warning(name):
    message = (
        f"The basis function {name} is not in use by the current "
        "baseline scheduler and may be deprecated shortly. "
        "Please contact the rubin_scheduler maintainers if "
        "this is in use elsewhere."
    )
    warnings.warn(message, FutureWarning)


class BaseBasisFunction:
    """Class that takes features and computes a reward function when
    called."""

    def __init__(self, nside=DEFAULT_NSIDE, bandname=None, filtername=None, **kwargs):
        # Set if basis function needs to be recalculated if there is a new
        # observation
        self.update_on_newobs = True
        # Set if basis function needs to be recalculated if conditions
        # change
        self.update_on_mjd = True
        # Dict to hold all the features we want to track
        self.survey_features = {}
        # Keep track of the last time the basis function was called.
        # If mjd doesn't change, use cached value
        self.mjd_last = None
        self.value = 0
        # list the attributes to compare to check if basis functions
        # are equal.
        self.attrs_to_compare = []
        # Do we need to recalculate the basis function
        self.recalc = True
        # Basis functions don't technically all need an nside, but so
        # many do might as well set it here
        self.nside = nside

        if filtername is not None:
            warnings.warn(
                "Use of `filtername` will be deprecated in favor of `bandname` at v4", FutureWarning
            )
            bandname = filtername
            # Save filtername as a backup in case someone tries to access it
            self.filtername = filtername
        self.bandname = bandname

    def add_observations_array(self, observations_array, observations_hpid):
        """Similar to add_observation, but for loading a whole
        array of observations at a time.

        Parameters
        ----------
        observations_array_in : `np.array`
            An array of completed observations (with columns like
            rubin_scheduler.scheduler.utils.ObservationArray).
            Should be sorted by MJD.
        observations_hpid_in : `np.array`
            Same as observations_array_in, but larger and with an
            additional column for HEALpix id.
            Each observation is listed multiple times,
            once for every HEALpix it overlaps.
        """

        for feature in self.survey_features:
            self.survey_features[feature].add_observations_array(observations_array, observations_hpid)
        if self.update_on_newobs:
            self.recalc = True

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        observation : `np.array`
            An array with information about the input observation
        indx : `np.array`
            The indices of the healpix map that the observation overlaps
            with
        """
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation, indx=indx)
        if self.update_on_newobs:
            self.recalc = True

    def check_feasibility(self, conditions):
        """If there is logic to decide if something is feasible
        (e.g., only if  moon is down), it can be calculated here.

        Helps prevent full __call__ from being called more than needed.
        """
        return True

    def _calc_value(self, conditions, **kwargs):
        self.value = 0
        # Update the last time we had an mjd
        self.mjd_last = conditions.mjd + 0
        self.recalc = False
        return self.value

    def __eq__(self):
        # XXX--to work on if we need to make a registry of basis functions.
        pass

    def __ne__(self):
        pass

    def __call__(self, conditions, **kwargs):
        """
        Parameters
        ----------
        conditions : `rubin_scheduler.scheduler.features.conditions` object
             Object that has attributes for all the current conditions.

        Return a reward healpix map or a reward scalar.
        """
        # If we are not feasible, return -inf
        if not self.check_feasibility(conditions):
            return -np.inf
        if self.recalc:
            self.value = self._calc_value(conditions, **kwargs)
        if self.update_on_mjd:
            if conditions.mjd != self.mjd_last:
                self.value = self._calc_value(conditions, **kwargs)
        return self.value

    def label(self):
        """Create a label for this basis function.

        Returns
        -------
        label : `str`
            A string suitable for labeling the basis function in a
            plot or table.
        """
        label = self.__class__.__name__.replace("BasisFunction", "")

        if self.bandname is not None:
            label += f" {self.bandname}"

        label += f" @{id(self)}"

        return label


class HealpixLimitedBasisFunctionMixin:
    """A mixin to limit a basis function to a set of Healpix pixels."""

    def __init__(self, hpid, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpid = hpid

    def check_feasibility(self, conditions):
        """Check the feasibility of the current set of conditions.

        Parameters
        ----------
        conditions : `rubin_scheduler.scheduler.features.Conditions`
            The conditions for which to test feasibility.

        Returns
        -------
        feasibility : `bool`
            True if the current set of conditions is feasible, False
            otherwise.
        """

        if super().check_feasibility(conditions):
            if self.recalc or (self.update_on_mjd and conditions.mjd != self.mjd_last):
                value = self._calc_value(conditions)
            else:
                value = super().value

            feasibility = np.nanmax(value) > -np.inf
        else:
            feasibility = False
        return feasibility

    def _calc_value(self, conditions, all_sky=False, **kwargs):
        all_sky_value = super()._calc_value(conditions, **kwargs)

        if all_sky:
            return all_sky_value

        if np.isscalar(all_sky_value):
            value = all_sky_value
        else:
            assert len(all_sky_value) == hp.nside2npix(self.nside)
            value = all_sky_value[self.hpid]

        return value


class ConstantBasisFunction(BaseBasisFunction):
    """Just add a constant"""

    def __call__(self, conditions, **kwargs):
        return 1


class SimpleArrayBasisFunction(BaseBasisFunction):
    def __init__(self, value, *args, **kwargs):
        self.assigned_value = value
        super().__init__(*args, **kwargs)

    def _calc_value(self, conditions, **kwargs):
        self.value = self.assigned_value
        return self.value


class DelayStartBasisFunction(BaseBasisFunction):
    """Force things to not run before a given night.

    Parameters
    ----------
    nights_delay : `float`, optional
        Return False until conditions.night >= nights_delay.
    """

    def __init__(self, nights_delay=365.25 * 5):
        super().__init__()
        self.nights_delay = nights_delay

    def check_feasibility(self, conditions):
        result = True
        if conditions.night < self.nights_delay:
            result = False
        return result


class BandDistBasisFunction(BaseBasisFunction):
    """Track band distribution, increase reward as fraction of observations
    in specified band drops.
    """

    def __init__(self, bandname="r"):
        super(BandDistBasisFunction, self).__init__(bandname=bandname)

        self.survey_features = {}
        # Count of all the observations
        self.survey_features["n_obs_count_all"] = features.NObsCount(bandname=None)
        # Count in band
        self.survey_features["n_obs_count_in_filt"] = features.NObsCount(bandname=bandname)

    def _calc_value(self, conditions, indx=None):
        result = self.survey_features["n_obs_count_all"].feature / (
            self.survey_features["n_obs_count_in_filt"].feature + 1
        )
        return result


class FilterDistBasisFunction(BandDistBasisFunction):
    """Deprecated version of BandDistBasisFunction"""

    def __init__(self, filtername="r"):
        warnings.warn("FilterDistBasisFunction deprecated for BandDistBasisFunction", FutureWarning)
        super().__init__(bandname=filtername)


class NObsPerYearBasisFunction(BaseBasisFunction):
    """Reward areas that have not been observed N-times in the last year

    Parameters
    ----------
    bandname : `str` ('r')
        The band to track
    footprint : `np.array`
        Should be a HEALpix map. Values of 0 or np.nan will be ignored.
    n_obs : `int` (3)
        The number of observations to demand
    season : `float` (300)
        The amount of time to allow pass before marking a region as "behind".
        Default 365.25 (days).
    season_start_hour : `float` (-2)
        When to start the season relative to RA 180 degrees away
        from the sun (hours)
    season_end_hour : `float` (2)
        When to consider a season ending, the RA relative to the sun + 180
        degrees. (hours)
    night_max : float (365)
        Set value to zero after night_max is reached (days)
    """

    def __init__(
        self,
        bandname="r",
        nside=DEFAULT_NSIDE,
        footprint=None,
        n_obs=3,
        season=300,
        season_start_hour=-4.0,
        season_end_hour=2.0,
        night_max=365,
        filtername=None,
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super(NObsPerYearBasisFunction, self).__init__(nside=nside, bandname=bandname)
        self.footprint = footprint
        self.n_obs = n_obs
        self.season = season
        self.season_start_hour = (season_start_hour) * np.pi / 12.0  # To radians
        self.season_end_hour = season_end_hour * np.pi / 12.0  # To radians

        self.survey_features["last_n_mjds"] = features.LastNObsTimes(
            nside=nside, bandname=bandname, n_obs=n_obs
        )
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.out_footprint = np.where((footprint == 0) | np.isnan(footprint))
        self.night_max = night_max

    def _calc_value(self, conditions, indx=None):
        if conditions.night > self.night_max:
            return 0

        result = self.result.copy()
        behind_pix = np.where((conditions.mjd - self.survey_features["last_n_mjds"].feature[0]) > self.season)
        result[behind_pix] = 1

        # let's ramp up the weight depending on how far into the
        # observing season the healpix is
        mid_season_ra = (conditions.sun_ra + np.pi) % (2.0 * np.pi)
        # relative RA
        relative_ra = (conditions.ra - mid_season_ra) % (2.0 * np.pi)
        relative_ra = (self.season_end_hour - relative_ra) % (2.0 * np.pi)
        # ok, now
        relative_ra[
            np.where(IntRounded(relative_ra) > IntRounded(self.season_end_hour - self.season_start_hour))
        ] = 0

        weight = relative_ra / (self.season_end_hour - self.season_start_hour)
        result *= weight

        # mask off anything outside the footprint
        result[self.out_footprint] = 0

        return result


class NGoodSeeingBasisFunction(BaseBasisFunction):
    """Try to get N "good seeing" images each observing season.

    Parameters
    ----------
    bandname : `str`
        Bandpass in which to count images. Default r.
    nside : `int`
        The nside of the map for the basis function. Should match
        survey and scheduler nside.
        Default None uses `set_default_nside`.
    seeing_fwhm_max : `float`
        Value to consider as "good" threshold (arcsec).
        Default of 0.8 arcseconds.
    m5_penalty_max : `float`
        The maximum depth loss that is considered acceptable (magnitudes),
        compared to the dark-sky map in this band.
        Default 0.5 magnitudes.
    n_obs_desired : `int`
        Number of good seeing observations to collect per season.
        Default 3.
    mjd_start : `float`
        The starting MJD of the survey.
        Default None uses `rubin_scheduler.utils.SURVEY_START_MJD`.
    footprint : `np.array`, (N,)
        Only use area where footprint > 0. Should be a HEALpix map.
        Default None calls `get_current_footprint()`.
    """

    def __init__(
        self,
        bandname="r",
        nside=DEFAULT_NSIDE,
        seeing_fwhm_max=0.8,
        m5_penalty_max=0.5,
        n_obs_desired=3,
        mjd_start=None,
        footprint=None,
        filtername=None,
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super().__init__(nside=nside, bandname=bandname)
        self.seeing_fwhm_max = seeing_fwhm_max
        self.m5_penalty_max = m5_penalty_max
        self.n_obs_desired = n_obs_desired
        if mjd_start is None:
            mjd_start = SURVEY_START_MJD
        self.mjd_start = mjd_start
        self.survey_features["N_good_seeing"] = features.NObservationsCurrentSeason(
            bandname=self.bandname,
            mjd_start=self.mjd_start,
            seeing_fwhm_max=self.seeing_fwhm_max,
            m5_penalty_max=self.m5_penalty_max,
            nside=self.nside,
        )
        # Set footprint to current survey footprint class if undefined.
        if footprint is None:
            footprints, labels = get_current_footprint(self.nside)
            footprint = footprints[self.bandname]
        self.footprint = footprint
        self.result = np.zeros(hp.nside2npix(self.nside))
        self.dark_map = None

    def _calc_value(self, conditions, indx=None):
        if self.bandname is not None:
            if self.dark_map is None:
                self.dark_map = dark_m5(
                    conditions.dec, self.bandname, conditions.site.latitude_rad, fiducial_FWHMEff=0.7
                )
        # Return the same kind of array (not float) regardless
        # of result
        result = self.result.copy()
        # Update the feature to the current time.
        self.survey_features["N_good_seeing"].season_update(conditions=conditions)

        m5_penalty = self.dark_map - conditions.m5_depth[self.bandname]
        potential_pixels = np.where(
            (m5_penalty <= self.m5_penalty_max)
            & (conditions.fwhm_eff[self.bandname] <= self.seeing_fwhm_max)
            & (self.survey_features["N_good_seeing"].feature < self.n_obs_desired)
            & (self.footprint > 0)
        )[0]

        result[potential_pixels] = 1
        return result


def az_rel_point(azs, point_az):
    az_rel_moon = (azs - point_az) % (2.0 * np.pi)
    if isinstance(azs, np.ndarray):
        over = np.where(az_rel_moon > np.pi)
        az_rel_moon[over] = 2.0 * np.pi - az_rel_moon[over]
    else:
        if az_rel_moon > np.pi:
            az_rel_moon = 2.0 * np.pi - az_rel_moon
    return az_rel_moon


class EclipticBasisFunction(BaseBasisFunction):
    """Mark the area around the ecliptic"""

    def __init__(self, nside=DEFAULT_NSIDE, distance_to_eclip=25.0):
        super(EclipticBasisFunction, self).__init__(nside=nside)
        self.distance_to_eclip = np.radians(distance_to_eclip)
        ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(self.nside)))
        self.result = np.zeros(ra.size)
        coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
        eclip_lat = coord.barycentrictrueecliptic.lat.radian
        good = np.where(np.abs(eclip_lat) < self.distance_to_eclip)
        self.result[good] += 1

    def __call__(self, conditions, indx=None):
        return self.result


class CadenceInSeasonBasisFunction(BaseBasisFunction):
    """Drive observations at least every N days in a given area

    Parameters
    ----------
    drive_map : `np.ndarray`, (N,)
        A HEALpix map with values of 1 where the cadence should be driven.
    bandname : `str`
        The bands that can count.
    season_span : `float`
        How long to consider a spot "in_season" (hours).
    cadence : `float`
        How long to wait before activating the basis function (days).
    """

    def __init__(
        self, drive_map, bandname="griz", season_span=2.5, cadence=2.5, nside=DEFAULT_NSIDE, filtername=None
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super(CadenceInSeasonBasisFunction, self).__init__(nside=nside, bandname=bandname)
        self.drive_map = drive_map
        self.season_span = season_span / 12.0 * np.pi  # To radians
        self.cadence = cadence
        self.survey_features["last_observed"] = features.LastObserved(nside=nside, bandname=bandname)
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        ra_mid_season = (conditions.sunRA + np.pi) % (2.0 * np.pi)

        angle_to_mid_season = np.abs(conditions.ra - ra_mid_season)
        over = np.where(IntRounded(angle_to_mid_season) > IntRounded(np.pi))
        angle_to_mid_season[over] = 2.0 * np.pi - angle_to_mid_season[over]

        days_lag = conditions.mjd - self.survey_features["last_observed"].feature

        active_pix = np.where(
            (IntRounded(days_lag) >= IntRounded(self.cadence))
            & (self.drive_map == 1)
            & (IntRounded(angle_to_mid_season) < IntRounded(self.season_span))
        )
        result[active_pix] = 1.0

        return result


class SeasonCoverageBasisFunction(BaseBasisFunction):
    """Basis function to encourage N observations per observing season.

    Parameters
    ----------
    bandname : `str`, optional
        Count observations in this band. Default 'r'.
    nside : `int`, optional
        Nside for the healpix map to use for the feature.
        This should match the nside of the survey and scheduler.
    footprint : `np.array` (N,), optional
        Healpix map of the footprint where one should demand coverage
        every season. Default None will call `get_current_footprint()`.
    n_per_season : `int`, optional
        The number of observations to attempt to gather every season.
        Default of 3 is suitable for first year template building.
    mjd_start : `float`, optional
        The mjd of the start of the survey (days).
        Default None uses `rubin_scheduler.utils.SURVEY_START_MJD`.
    season_frac_start : `float`
        Only start trying to gather observations after a season
        is fractionally this far along.
        Seasons start when the apparent position of sun is at the RA of
        the pixel (0) and finish when the sun returns again to this RA.
        The default of 0.5 means that the basis function will not
        start returning values until the RA reaches the peak of its season.
    """

    def __init__(
        self,
        bandname="r",
        nside=DEFAULT_NSIDE,
        footprint=None,
        n_per_season=3,
        mjd_start=None,
        season_frac_start=0.5,
        filtername=None,
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        send_unused_deprecation_warning("SeasonCoverageBasisFunction")
        super().__init__(nside=nside, bandname=bandname)

        if footprint is None:
            footprints, labels = get_current_footprint(self.nside)
            footprint = footprints[self.bandname]
        self.footprint = footprint
        # Calculate the RA values for each spot on the footprint
        ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
        self.ra_deg = np.degrees(ra)

        self.n_per_season = n_per_season
        if mjd_start is None:
            mjd_start = SURVEY_START_MJD
        self.mjd_start = mjd_start
        self.season_frac_start = season_frac_start
        # Track how many observations have been taken at each RA/Dec
        # in the current observing season (for that point on the sky).
        self.survey_features["n_obs_season"] = features.NObservationsCurrentSeason(
            bandname=bandname, nside=nside, mjd_start=mjd_start
        )
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        # Update the feature to the current time
        self.survey_features["n_obs_season"].season_update(conditions=conditions)
        feature = self.survey_features["n_obs_season"].feature

        # Get the season from the conditions object.
        season = conditions.season

        # Evaluate where there are not yet enough observations and also
        # that season is far enough along to require more weight.
        not_enough = np.where(
            (self.footprint > 0)
            & (feature < self.n_per_season)
            & ((IntRounded(season - np.floor(season)) > IntRounded(self.season_frac_start)))
        )
        result[not_enough] = 1
        return result


class AvoidFastRevisitsBasisFunction(BaseBasisFunction):
    """Marks targets as unseen if they are in a specified time window
    in order to avoid fast revisits.

    Parameters
    ----------
    bandname : `str` or None
        The name of the band for this target map.
        Using None will match visits in any band.
    gap_min : `float`
        Minimum time for the gap (minutes).
    nside: `int` or None
        The healpix resolution.
    penalty_val : `float`
        The reward value to use for regions to penalize.
        Will be masked if set to np.nan (default).
    """

    def __init__(self, bandname="r", nside=DEFAULT_NSIDE, gap_min=25.0, penalty_val=np.nan, filtername=None):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super().__init__(nside=nside, bandname=bandname)

        self.bandname = bandname
        self.penalty_val = penalty_val

        self.gap_min = IntRounded(gap_min / 60.0 / 24.0)
        self.nside = nside

        self.survey_features = dict()
        self.survey_features["Last_observed"] = features.LastObserved(bandname=bandname, nside=nside, fill=0)

    def _calc_value(self, conditions, indx=None):
        result = np.ones(hp.nside2npix(self.nside), dtype=float)
        diff = IntRounded(conditions.mjd - self.survey_features["Last_observed"].feature)
        bad = np.where(diff < self.gap_min)[0]
        result[bad] = self.penalty_val
        return result


class NearSunHighAirmassBasisFunction(BaseBasisFunction):
    """Reward areas on the sky at high airmass, within 90 degrees azimuth
    of the Sun, such as suitable for the near-sun twilight microsurvey for
    near- or interior-to earth asteroids.

    Parameters
    ----------
    nside : `int`, optional
        Nside for the basis function. If None, uses `set_default_nside()`.
    max_airmass : `float`, oprionl
        The maximum airmass to try and observe (unitless).
    penalty : `float`, optional
        The value to fill in non-rewarded parts of the sky.
        Default np.nan, which serves to mask regions exceeding the airmass
        limit and more than 90 degrees azimuth toward the sun.
    """

    def __init__(self, nside=DEFAULT_NSIDE, max_airmass=2.5, penalty=np.nan):
        super().__init__(nside=nside)
        self.max_airmass = IntRounded(max_airmass)
        self.result = np.empty(hp.nside2npix(self.nside))
        self.result.fill(penalty)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        valid_airmass = np.where(np.isfinite(conditions.airmass))[0]
        good_pix = np.where(
            (conditions.airmass[valid_airmass] >= 1.0)
            & (IntRounded(conditions.airmass[valid_airmass]) < self.max_airmass)
            & (IntRounded(np.abs(conditions.az_to_sun[valid_airmass])) < IntRounded(np.pi / 2.0))
        )
        result[valid_airmass[good_pix]] = (
            conditions.airmass[valid_airmass][good_pix] / self.max_airmass.initial
        )
        return result


class VisitRepeatBasisFunction(BaseBasisFunction):
    """
    Basis function to reward re-visiting an area on the sky.
    Looking for Solar System objects.

    Parameters
    ----------
    gap_min : `float` (15.)
        Minimum time for the gap (minutes)
    gap_max : `float` (45.)
        Maximum time for a gap
    bandname : `str` ('r')
        The band(s) to count with pairs
    npairs : `int` (1)
        The number of pairs of observations to attempt to gather
    """

    def __init__(
        self, gap_min=25.0, gap_max=45.0, bandname="r", nside=DEFAULT_NSIDE, npairs=1, filtername=None
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super(VisitRepeatBasisFunction, self).__init__(nside=nside, bandname=bandname)

        self.gap_min = IntRounded(gap_min / 60.0 / 24.0)
        self.gap_max = IntRounded(gap_max / 60.0 / 24.0)
        self.npairs = npairs
        self.survey_features = {}
        # Track the number of pairs that have been taken in a night
        self.survey_features["Pair_in_night"] = features.PairInNight(
            bandname=bandname, gap_min=gap_min, gap_max=gap_max, nside=nside
        )

        # When was it last observed
        # XXX--since this feature is also in Pair_in_night, I should just
        # access that one!
        self.survey_features["Last_observed"] = features.LastObserved(bandname=bandname, nside=nside)

    def _calc_value(self, conditions, indx=None):
        result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        if indx is None:
            indx = np.arange(result.size)
        diff = conditions.mjd - self.survey_features["Last_observed"].feature[indx]
        mask = np.isnan(diff)
        # remove NaNs from diff, but save mask so we exclude those values
        # later.
        diff[mask] = 0.0
        ir_diff = IntRounded(diff)
        good = np.where(
            (ir_diff >= self.gap_min)
            & (ir_diff <= self.gap_max)
            & (self.survey_features["Pair_in_night"].feature[indx] < self.npairs)
            & (~mask)
        )[0]
        result[indx[good]] += 1.0
        return result


class M5DiffBasisFunction(BaseBasisFunction):
    """Basis function based on the 5-sigma depth.
    Look up the best depth a healpixel achieves, and compute
    the limiting depth difference given current conditions

    Parameters
    ----------
    bandname : `str`, optional
        The band to consider for visits.
    fiducial_FWHMEff : `float`, optional
        The zenith seeing to assume for "good" conditions.
        While the dark sky depth map simply scales with this value,
        picking a reasonable fiducial_FWHMEff is important because
        this effects the overall value and scale of the reward
        from the basis function.
    nside : `int`, optional
        The nside for the basis function.
        Default None uses `set_default_nside()`.
    apply_cloud_extinction : `bool`
        Apply extinction from cloud maps. Default False.
    """

    def __init__(
        self,
        bandname="r",
        fiducial_FWHMEff=0.7,
        nside=DEFAULT_NSIDE,
        filtername=None,
        apply_cloud_extinction=False,
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super().__init__(nside=nside, bandname=bandname)
        # The dark sky surface brightness values
        self.dark_map = None
        self.fiducial_FWHMEff = fiducial_FWHMEff
        self.bandname = bandname
        self.apply_cloud_extinction = apply_cloud_extinction

    def _calc_value(self, conditions, indx=None):
        if self.dark_map is None:
            self.dark_map = dark_m5(
                conditions.dec, self.bandname, conditions.site.latitude_rad, self.fiducial_FWHMEff
            )

        result = conditions.m5_depth[self.bandname] - self.dark_map
        if self.apply_cloud_extinction:
            if conditions.cloud_maps is not None:
                extinction = conditions.cloud_maps.extinction_closest(conditions.mjd)
                result -= extinction
        return result


class M5DiffAtHpixBasisFunction(HealpixLimitedBasisFunctionMixin, M5DiffBasisFunction):
    pass


class StrictBandBasisFunction(BaseBasisFunction):
    """Remove the bonus for staying in the same band
    if certain conditions are met.

    If the moon rises/sets or twilight starts/ends, it makes a lot of sense
    to consider a band change. This basis function rewards if it matches
    the current band, the moon rises or sets, twilight starts or stops,
    or there has been a large gap since the last observation.

    Parameters
    ----------
    time_lag : `float` (10.)
        If there is a gap between observations longer than this,
        let the band change (minutes)
    twi_change : `float` (-18.)
        The sun altitude to consider twilight starting/ending (degrees)
    note_free : `str` ('DD')
        No penalty for changing bands if the last observation note field
        includes `note_free` string.
        Useful for giving a free band change after deep drilling sequence
    """

    def __init__(self, time_lag=10.0, bandname="r", twi_change=-18.0, note_free="DD"):
        super(StrictBandBasisFunction, self).__init__(bandname=bandname)

        self.time_lag = time_lag / 60.0 / 24.0  # Convert to days
        self.twi_change = np.radians(twi_change)

        self.survey_features = {}
        self.survey_features["Last_observation"] = features.LastObservation()
        self.note_free = note_free

    def _calc_value(self, conditions, **kwargs):
        # Did the moon set or rise since last observation?
        moon_changed = conditions.moon_alt * self.survey_features["Last_observation"].feature["moonAlt"] < 0

        # Are we already in the band (or at start of night)?
        in_band = (conditions.current_band == self.bandname) | (conditions.current_band is None)

        # Has enough time past?
        time_past = IntRounded(
            conditions.mjd - self.survey_features["Last_observation"].feature["mjd"]
        ) > IntRounded(self.time_lag)

        # Did twilight start/end?
        twi_changed = (conditions.sun_alt - self.twi_change) * (
            self.survey_features["Last_observation"].feature["sunAlt"] - self.twi_change
        ) < 0

        # Did we just finish a DD sequence
        was_dd = self.note_free in self.survey_features["Last_observation"].feature["scheduler_note"]

        # Is the band mounted?
        mounted = self.bandname in conditions.mounted_bands

        if (moon_changed | in_band | time_past | twi_changed | was_dd) & mounted:
            result = 1.0
        else:
            result = 0.0

        return result


class StrictFilterBasisFunction(StrictBandBasisFunction):
    """Deprecated in favor of StrictBandBasisFunction"""

    def __init__(self, time_lag=10.0, filtername="r", twi_change=-18.0, note_free="DD"):
        warnings.warn(
            "StrictFilterBasisFunction deprecated in favor of StrictBandBasisFunction", FutureWarning
        )
        super().__init__(time_lag=time_lag, bandname=filtername, twi_change=twi_change, note_free=note_free)


class BandChangeBasisFunction(BaseBasisFunction):
    """Reward staying in the current band."""

    def __init__(self, bandname="r"):
        super(BandChangeBasisFunction, self).__init__(bandname=bandname)

    def _calc_value(self, conditions, **kwargs):
        if (conditions.current_band == self.bandname) | (conditions.current_band is None):
            result = 1.0
        else:
            result = 0.0
        return result


class FilterChangeBasisFunction(BandChangeBasisFunction):
    """Deprecated in favor of BandChangeBasisFunction"""

    def __init__(self, filtername="r"):
        warnings.warn(
            "FilterChangeBasisFunction deprecated in favor of BandChangeBasisFunction", FutureWarning
        )
        super().__init__(bandname=filtername)


class SlewtimeBasisFunction(BaseBasisFunction):
    """Reward slews that take little time

    Parameters
    ----------
    max_time : `float`
         The estimated maximum slewtime (seconds).
         Used to normalize so the basis function spans ~ -1-0
         in reward units. Default 135 seconds corresponds to just
         slightly less than a band change.
    bandname : `str` or None, optional
        The band to check for pre-post slewtime estimates.
        If the band is None, then bandpasses changes are NOT considered
        when calculating slewtime.
        If a slew includes a band change, other basis functions will
        decide on the reward, so the result here can be 0.
    nside : `int`, optional
        Nside for the basis function.
        Default None will use `set_default_nside()`.
    """

    def __init__(self, max_time=135.0, bandname="r", nside=DEFAULT_NSIDE, filtername=None):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super(SlewtimeBasisFunction, self).__init__(nside=nside, bandname=bandname)

        self.maxtime = max_time
        self.nside = nside
        self.bandname = bandname

    def _calc_value(self, conditions, indx=None):
        # If we are in a different band, the
        # BandChangeBasisFunction will take it
        # But we can still use the MASK returned by
        # the slewtime map to remove inaccessible parts of the sky
        if conditions.current_band != self.bandname and self.bandname is not None:
            if np.size(conditions.slewtime) > 1:
                result = np.where(np.isfinite(conditions.slewtime), 0, np.nan)
            else:
                result = 0
        else:
            # Need to make sure smaller slewtime is larger reward.
            if np.size(conditions.slewtime) > 1:
                # Slewtime map can contain nans and/or
                # infs - mask these with nans
                result = np.where(
                    np.isfinite(conditions.slewtime),
                    -conditions.slewtime / self.maxtime,
                    np.nan,
                )
            else:
                result = -conditions.slewtime / self.maxtime
        return result


class CadenceEnhanceBasisFunction(BaseBasisFunction):
    """Drive a certain cadence

    Parameters
    ----------
    bandname : `str` ('gri')
        The band(s) that should be grouped together
    supress_window : `list` of `float`
        The start and stop window for when observations should be repressed
        (days)
    apply_area : healpix map
        The area over which to try and drive the cadence.
        Good values as 1, no cadence drive 0.
        Probably works as a bool array too.
    """

    def __init__(
        self,
        bandname="gri",
        nside=DEFAULT_NSIDE,
        supress_window=[0, 1.8],
        supress_val=-0.5,
        enhance_window=[2.1, 3.2],
        enhance_val=1.0,
        apply_area=None,
        filtername=None,
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super(CadenceEnhanceBasisFunction, self).__init__(nside=nside, bandname=bandname)

        self.supress_window = np.sort(supress_window)
        self.supress_val = supress_val
        self.enhance_window = np.sort(enhance_window)
        self.enhance_val = enhance_val

        self.survey_features = {}
        self.survey_features["last_observed"] = features.LastObserved(bandname=bandname)

        self.empty = np.zeros(hp.nside2npix(self.nside), dtype=float)
        # No map, try to drive the whole area
        if apply_area is None:
            self.apply_indx = np.arange(self.empty.size)
        else:
            self.apply_indx = np.where(apply_area != 0)[0]

    def _calc_value(self, conditions, indx=None):
        # copy an empty array
        result = self.empty.copy()
        if indx is not None:
            ind = np.intersect1d(indx, self.apply_indx)
        else:
            ind = self.apply_indx
        if np.size(ind) == 0:
            result = 0
        else:
            mjd_diff = conditions.mjd - self.survey_features["last_observed"].feature[ind]
            to_supress = np.where(
                (IntRounded(mjd_diff) > IntRounded(self.supress_window[0]))
                & (IntRounded(mjd_diff) < IntRounded(self.supress_window[1]))
            )
            result[ind[to_supress]] = self.supress_val
            to_enhance = np.where(
                (IntRounded(mjd_diff) > IntRounded(self.enhance_window[0]))
                & (IntRounded(mjd_diff) < IntRounded(self.enhance_window[1]))
            )
            result[ind[to_enhance]] = self.enhance_val
        return result


# https://docs.astropy.org/en/stable/_modules/astropy/modeling
# functional_models.html#Trapezoid1D
def trapezoid(x, amplitude, x_0, width, slope):
    """One dimensional Trapezoid model function"""
    # Compute the four points where the trapezoid changes slope
    # x1 <= x2 <= x3 <= x4
    x2 = x_0 - width / 2.0
    x3 = x_0 + width / 2.0
    x1 = x2 - amplitude / slope
    x4 = x3 + amplitude / slope

    result = x * 0

    # Compute model values in pieces between the change points
    range_a = np.logical_and(x >= x1, x < x2)
    range_b = np.logical_and(x >= x2, x < x3)
    range_c = np.logical_and(x >= x3, x < x4)

    result[range_a] = slope * (x[range_a] - x1)
    result[range_b] = amplitude
    result[range_c] = slope * (x4 - x[range_c])

    return result


class CadenceEnhanceTrapezoidBasisFunction(BaseBasisFunction):
    """Drive a certain cadence, like CadenceEnhanceBasisFunction
    but with smooth transitions

    Parameters
    ----------
    bandname : `str` ('gri')
        The band(s) that should be grouped together

    XXX--fill out doc string!
    """

    def __init__(
        self,
        bandname="gri",
        nside=DEFAULT_NSIDE,
        delay_width=2,
        delay_slope=2.0,
        delay_peak=0,
        delay_amp=0.5,
        enhance_width=3.0,
        enhance_slope=2.0,
        enhance_peak=4.0,
        enhance_amp=1.0,
        apply_area=None,
        season_limit=None,
        filtername=None,
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super(CadenceEnhanceTrapezoidBasisFunction, self).__init__(nside=nside, bandname=bandname)

        self.delay_width = delay_width
        self.delay_slope = delay_slope
        self.delay_peak = delay_peak
        self.delay_amp = delay_amp
        self.enhance_width = enhance_width
        self.enhance_slope = enhance_slope
        self.enhance_peak = enhance_peak
        self.enhance_amp = enhance_amp

        self.season_limit = season_limit / 12 * np.pi  # To radians

        self.survey_features = {}
        self.survey_features["last_observed"] = features.LastObserved(bandname=bandname)

        self.empty = np.zeros(hp.nside2npix(self.nside), dtype=float)
        # No map, try to drive the whole area
        if apply_area is None:
            self.apply_indx = np.arange(self.empty.size)
        else:
            self.apply_indx = np.where(apply_area != 0)[0]

    def suppress_enhance(self, x):
        result = x * 0
        result -= trapezoid(x, self.delay_amp, self.delay_peak, self.delay_width, self.delay_slope)
        result += trapezoid(
            x,
            self.enhance_amp,
            self.enhance_peak,
            self.enhance_width,
            self.enhance_slope,
        )

        return result

    def season_len(self, conditions):
        ra_mid_season = (conditions.sunRA + np.pi) % (2.0 * np.pi)
        angle_to_mid_season = np.abs(conditions.ra - ra_mid_season)
        over = np.where(IntRounded(angle_to_mid_season) > IntRounded(np.pi))
        angle_to_mid_season[over] = 2.0 * np.pi - angle_to_mid_season[over]

        return angle_to_mid_season

    def _calc_value(self, conditions, indx=None):
        # copy an empty array
        result = self.empty.copy()
        if indx is not None:
            ind = np.intersect1d(indx, self.apply_indx)
        else:
            ind = self.apply_indx
        if np.size(ind) == 0:
            result = 0
        else:
            mjd_diff = conditions.mjd - self.survey_features["last_observed"].feature[ind]
            result[ind] += self.suppress_enhance(mjd_diff)

        if self.season_limit is not None:
            radians_to_midseason = self.season_len(conditions)
            outside_season = np.where(radians_to_midseason > self.season_limit)
            result[outside_season] = 0

        return result


class AzimuthBasisFunction(BaseBasisFunction):
    """Reward staying in the same azimuth range.
    Possibly better than using slewtime, especially when selecting a
    large area of sky.
    """

    def __init__(self, nside=DEFAULT_NSIDE):
        send_unused_deprecation_warning("AzimuthBasisFunction")
        super(AzimuthBasisFunction, self).__init__(nside=nside)

    def _calc_value(self, conditions, indx=None):
        az_dist = conditions.az - conditions.tel_az
        az_dist = az_dist % (2.0 * np.pi)
        over = np.where(az_dist > np.pi)
        az_dist[over] = 2.0 * np.pi - az_dist[over]
        # Normalize sp between 0 and 1
        result = az_dist / np.pi
        return result


class AzModuloBasisFunction(BaseBasisFunction):
    """Try to replicate the Rothchild et al cadence forcing by
    only observing on limited az ranges per night.

    Parameters
    ----------
    az_limits : `list` of `float` pairs (None)
        The azimuth limits (degrees) to use.
    """

    def __init__(self, nside=DEFAULT_NSIDE, az_limits=None, out_of_bounds_val=-1.0):
        super(AzModuloBasisFunction, self).__init__(nside=nside)
        self.result = np.ones(hp.nside2npix(self.nside))
        if az_limits is None:
            spread = 100.0 / 2.0
            self.az_limits = np.radians(
                [
                    [360 - spread, spread],
                    [90.0 - spread, 90.0 + spread],
                    [180.0 - spread, 180.0 + spread],
                ]
            )
        else:
            self.az_limits = np.radians(az_limits)
        self.mod_val = len(self.az_limits)
        self.out_of_bounds_val = out_of_bounds_val

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        az_lim = self.az_limits[np.max(conditions.night) % self.mod_val]

        if az_lim[0] < az_lim[1]:
            out_pix = np.where(
                (IntRounded(conditions.az) < IntRounded(az_lim[0]))
                | (IntRounded(conditions.az) > IntRounded(az_lim[1]))
            )
        else:
            out_pix = np.where(
                (IntRounded(conditions.az) < IntRounded(az_lim[0]))
                | (IntRounded(conditions.az) > IntRounded(az_lim[1]))
            )[0]
        result[out_pix] = self.out_of_bounds_val
        return result


class DecModuloBasisFunction(BaseBasisFunction):
    """Emphasize dec bands on a nightly varying basis

    Parameters
    ----------
    dec_limits : `list` of `float` pairs (None)
        The azimuth limits (degrees) to use.
    """

    def __init__(self, nside=DEFAULT_NSIDE, dec_limits=None, out_of_bounds_val=-1.0):
        super(DecModuloBasisFunction, self).__init__(nside=nside)

        npix = hp.nside2npix(nside)
        hpids = np.arange(npix)
        ra, dec = _hpid2_ra_dec(nside, hpids)

        self.results = []

        if dec_limits is None:
            self.dec_limits = np.radians([[-90.0, -32.8], [-32.8, -12.0], [-12.0, 35.0]])
        else:
            self.dec_limits = np.radians(dec_limits)
        self.mod_val = len(self.dec_limits)
        self.out_of_bounds_val = out_of_bounds_val

        for limits in self.dec_limits:
            good = np.where((dec >= limits[0]) & (dec < limits[1]))[0]
            tmp = np.zeros(npix)
            tmp[good] = 1
            self.results.append(tmp)

    def _calc_value(self, conditions, indx=None):
        night_index = np.max(conditions.night % self.mod_val)
        result = self.results[night_index]

        return result


class MapModuloBasisFunction(BaseBasisFunction):
    """Similar to Dec_modulo, but now use input masks

    Parameters
    ----------
    inmaps : `list` of hp arrays
    """

    def __init__(self, inmaps):
        nside = hp.npix2nside(np.size(inmaps[0]))
        super(MapModuloBasisFunction, self).__init__(nside=nside)
        self.maps = inmaps
        self.mod_val = len(inmaps)

    def _calc_value(self, conditions, indx=None):
        indx = np.max(conditions.night % self.mod_val)
        result = self.maps[indx]
        return result


class VisitGap(BaseBasisFunction):
    """Basis function to create a visit gap based on the survey note field.

    Parameters
    ----------
    note : `str`
        Value of the observation "scheduler_note" field to be masked.
    band_names : list [str], optional
        List of band names that will be considered when evaluating
        if the gap has passed.
    gap_min : float (optional)
        Time gap (default=25, in minutes).
    penalty_val : float or np.nan
        Value of the penalty to apply (default is np.nan).

    Notes
    -----
    When a list of bands is provided, all bands must be observed before
    the gap requirement will be activated, and once activated, only
    observations in these bands will be evaluated in context of whether
    the last observation was at least gap in the past.
    """

    def __init__(self, note, band_names=None, gap_min=25.0, penalty_val=np.nan, filter_names=None):
        if filter_names is not None:
            warnings.warn("filter_names deprecated in favor of band_names", FutureWarning)
            band_names = filter_names
        super().__init__()
        self.penalty_val = penalty_val

        self.gap = gap_min / 60.0 / 24.0
        self.band_names = band_names

        self.survey_features = dict()
        if self.band_names is not None:
            for bandname in self.band_names:
                self.survey_features[f"LastObservationMjd::{bandname}"] = features.LastObservationMjd(
                    note=note, bandname=bandname
                )
        else:
            self.survey_features["LastObservationMjd"] = features.LastObservationMjd(scheduler_note=note)

    def check_feasibility(self, conditions):
        notes_last_observed = [last_observed.feature for last_observed in self.survey_features.values()]

        if any([last_observed is None for last_observed in notes_last_observed]):
            return True

        after_gap = [conditions.mjd - last_observed > self.gap for last_observed in notes_last_observed]

        return all(after_gap)

    def _calc_value(self, conditions, indx=None):
        return 1.0 if self.check_feasibility(conditions) else self.penalty_val


class AvoidDirectWind(BaseBasisFunction):
    """Mask the sky where the wind pressure exceeds `wind_speed_maximum`.

    Parameters
    ----------
    wind_speed_maximum : `float`, optional
        Wind speed to mark regions as unobservable (in m/s).
    nside : `int`, optional
        The nside for the basis function. Default None uses
        `set_default_nside()`.
    """

    def __init__(self, wind_speed_maximum=20.0, nside=DEFAULT_NSIDE):
        super().__init__(nside=nside)

        self.wind_speed_maximum = wind_speed_maximum

    def _calc_value(self, conditions, indx=None):
        reward_map = np.zeros(hp.nside2npix(self.nside))

        if conditions.wind_speed is None or conditions.wind_direction is None:
            return reward_map

        wind_pressure = conditions.wind_speed * np.cos(conditions.az - conditions.wind_direction)

        reward_map -= wind_pressure**2.0

        mask = wind_pressure > self.wind_speed_maximum

        reward_map[mask] = np.nan

        return reward_map


class BalanceVisits(BaseBasisFunction):
    """Balance visits across multiple surveys.

    Parameters
    ----------
    nobs_reference : `int`
        Expected number of observations across all interested surveys.
    note_survey : `str`
        Note value for the current survey.
    note_interest : `str`
        Substring with the name of interested surveys to be accounted.
    nside : `int`
        Healpix map resolution.

    Notes
    -----
    This basis function is designed to balance the reward of a group of
    surveys, such that the group get a reward boost based on the required
    collective number of observations.

    For example, if you have 3 surveys (e.g. SURVEY_A_REGION_1,
    SURVEY_A_REGION_2, SURVEY_A_REGION_3), when one of them is observed
    once (SURVEY_A_REGION_1) they all get a small reward boost proportional
    to the collective number of observations (`nobs_reference`). Further
    observations of SURVEY_A_REGION_1 would now cause the other surveys
    to gain a reward boost in relative to it.
    """

    def __init__(self, nobs_reference, note_survey, note_interest, nside=DEFAULT_NSIDE):
        super().__init__(nside=nside)

        self.nobs_reference = nobs_reference

        self.survey_features = {}
        self.survey_features["n_obs_survey"] = features.NObsCount(scheduler_note=note_survey)
        self.survey_features["n_obs_survey_interest"] = features.NObsCount(scheduler_note=note_interest)

    def _calc_value(self, conditions, indx=None):
        return (1 + np.floor(self.survey_features["n_obs_survey_interest"].feature / self.nobs_reference)) / (
            self.survey_features["n_obs_survey"].feature
            if self.survey_features["n_obs_survey"].feature > 0
            else 1
        )


class RewardNObsSequence(BaseBasisFunction):
    """Reward taking a sequence of observations.

    Parameters
    ----------
    n_obs_survey : `int`
        Number of observations to reward.
    note_survey : `str`
        The value of the observation note, to take into account.
    nside : `int`, optional
        Healpix map resolution (ignored).

    Notes
    -----
    This basis function is useful when a survey is composed of more than
    one observation (e.g. in different bands) and one wants to make sure
    they are all taken together.
    If the sequence is programmed into the FieldSurvey, this isn't necessary.
    """

    def __init__(self, n_obs_survey, note_survey, nside=DEFAULT_NSIDE):
        super().__init__(nside=nside)

        self.n_obs_survey = n_obs_survey

        self.survey_features = {}
        self.survey_features["n_obs_survey"] = features.NObsCount(scheduler_note=note_survey)

    def _calc_value(self, conditions, indx=None):
        return self.survey_features["n_obs_survey"].feature % self.n_obs_survey


class RewardRisingBasisFunction(BaseBasisFunction):
    """Reward parts of the sky that are rising.
    Optionally, mask out parts of the sky that are not rising.

    This produces a reward that increases
    as the field rises toward zenith, then abruptly
    falls as the field passes zenith.
    Negative hour angles (or hour angles > 180 degrees)
    indicate a rising point on the sky.

    Parameters
    ----------
    slope : `float`
        Sets the 'slope' of how fast the basis function
        value changes with hour angle.
    penalty_val : `float` or `np.nan`, optional
        Sets the value for the part of the sky which is
        not rising (hour angle between 0 and 180).
        Using a value of np.nan will mask this region of sky,
        a value of 0 will just make this non-rewarding.
    nside : `int` or None, optional
        Nside for the healpix map, default of None uses scheduler default.
    """

    def __init__(self, slope=0.1, penalty_val=0, nside=DEFAULT_NSIDE):
        super().__init__(nside=nside)
        self.slope = slope
        self.penalty_val = penalty_val

    # Probably not needed
    def check_feasibility(self, conditions):
        return True

    def _calc_value(self, conditions, indx=None):
        # HA should be available in the conditions object
        value = self.slope * conditions.HA
        past_zenith = np.where(conditions.HA < np.pi)
        value[past_zenith] = self.penalty_val
        return value
