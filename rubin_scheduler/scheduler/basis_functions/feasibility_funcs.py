__all__ = (
    "FilterLoadedBasisFunction",
    "BandLoadedBasisFunction",
    "OnceInNightBasisFunction",
    "SunAltHighLimitBasisFunction",
    "TimeToTwilightBasisFunction",
    "NotTwilightBasisFunction",
    "AfterEveningTwiBasisFunction",
    "ForceDelayBasisFunction",
    "HourAngleLimitBasisFunction",
    "MoonDownBasisFunction",
    "FractionOfObsBasisFunction",
    "CloudedOutBasisFunction",
    "RisingMoreBasisFunction",
    "SoftDelayBasisFunction",
    "LookAheadDdfBasisFunction",
    "SunAltLimitBasisFunction",
    "TimeInTwilightBasisFunction",
    "NightModuloBasisFunction",
    "EndOfEveningBasisFunction",
    "TimeToScheduledBasisFunction",
    "CloseToTwilightBasisFunction",
    "MoonDistPointRangeBasisFunction",
    "AirmassPointRangeBasisFunction",
    "InTimeWindowBasisFunction",
    "MoonAltLimitBasisFunction",
    "OnlyBeforeNightBasisFunction",
)

import warnings

import numpy as np

from rubin_scheduler.scheduler import features
from rubin_scheduler.scheduler.basis_functions import BaseBasisFunction
from rubin_scheduler.scheduler.utils import IntRounded, match_hp_resolution
from rubin_scheduler.utils import DEFAULT_NSIDE, _angular_separation, ra_dec2_hpid


def send_unused_deprecation_warning(name):
    message = (
        f"The feasibility basis function {name} is not in use, "
        "may be broken, and will be deprecated shortly. "
        "Please contact the rubin_scheduler maintainers if "
        "this is in use elsewhere."
    )
    warnings.warn(message, FutureWarning)


class OnlyBeforeNightBasisFunction(BaseBasisFunction):
    """Only return feasible if current night is less than
    maximum

    Parameters
    ----------
    night_max : `int`
        The maximum night. Default 366.
    """

    def __init__(self, night_max=366):
        super(OnlyBeforeNightBasisFunction, self).__init__()
        self.night_max = night_max

    def check_feasibility(self, conditions, indx=None):
        result = True
        if conditions.night > self.night_max:
            result = False
        return result


class BandLoadedBasisFunction(BaseBasisFunction):
    """Check that the band(s) needed are loaded.

    Are the bands in `bandnames` loaded and available?

    Parameters
    ----------
    bandnames : `str` or `list` [ `str` ]
        The bandnames that need to be mounted to execute.
    """

    def __init__(self, bandnames="r"):
        super(BandLoadedBasisFunction, self).__init__()
        if not isinstance(bandnames, list):
            bandnames = [bandnames]
        self.bandnames = bandnames

    def check_feasibility(self, conditions):
        for bandname in self.bandnames:
            result = bandname in conditions.mounted_bands
            if result is False:
                return result
        return result


class FilterLoadedBasisFunction(BandLoadedBasisFunction):
    def __init__(self, filternames="r"):
        warnings.warn("Deprecated in favor of BandLoadedBasisFunction", FutureWarning)
        super().__init__(bandnames=filternames)


class InTimeWindowBasisFunction(BaseBasisFunction):
    """Only let a survey go if it is in a defined season

    Parameters
    ----------
    mjd_windows : `list`
        List of mjd pairs. feasability will only pass if the
        current MJD falls in a viable range, e.g.
        seaons=[[1,2], [10,13]], means observations can happen
        on mjd=1,2,10,11,12,13.
    """

    def __init__(self, mjd_windows=[]):
        super().__init__()
        self.mjd_windows = mjd_windows

    def check_feasibility(self, conditions):
        result = False
        for mjd_windows in self.mjd_windows:
            if np.min(mjd_windows) <= conditions.mjd <= np.max(mjd_windows):
                result = True
        return result


class MoonAltLimitBasisFunction(BaseBasisFunction):
    """Only observe if the moon is below a given altitude limit.

    Parameters
    ----------
    alt_limit : `float`
        The maximum altitude for the moon. Default -5 (degrees)
    """

    def __init__(self, alt_limit=-5):
        super(MoonAltLimitBasisFunction, self).__init__()
        self.alt_limit = np.radians(alt_limit)

    def check_feasibility(self, conditions):
        result = True
        if conditions.moon_alt > self.alt_limit:
            result = False
        return result


class AirmassPointRangeBasisFunction(BaseBasisFunction):
    """Set an airmass limit for a single point
    Maps the point to the nearest HEALpixel and uses
    the Conditions airmass map, so precision of the
    airmass limit that is applied is limited by the
    HEALpix resolution.

    Parameters
    ----------
    ra : `float`
        The RA of the point (degrees)
    dec : `float`
        The Dec of the point (degrees)
    airmass_range : `list`
        The valid airmass range, default [1.05, 2.7].
    """

    def __init__(self, ra, dec, airmass_range=[1.05, 2.7], nside=DEFAULT_NSIDE):
        super().__init__()
        self.hpid = ra_dec2_hpid(nside, ra, dec)
        self.airmass_range = airmass_range

    def check_feasibility(self, conditions):
        result = False
        airmass = match_hp_resolution(conditions.airmass, nside_out=self.nside)[self.hpid]
        if (np.min(self.airmass_range) <= airmass) & (airmass <= np.max(self.airmass_range)):
            result = True
        return result


class MoonDistPointRangeBasisFunction(BaseBasisFunction):
    """Set a moon distance limit for a single point.

    Parameters
    ----------
    ra : `float`
        The RA of the point (degrees)
    dec : `float`
        The Dec of the point (degrees)
    moon_limit : `float`
        The angular distance to demand from the moon (degrees).
        Default 15.
    """

    def __init__(self, ra, dec, moon_limit=15.0):
        super().__init__()
        self.ra = np.radians(ra)
        self.dec = np.radians(dec)
        self.moon_limit = np.radians(moon_limit)

    def check_feasibility(self, conditions):
        result = False
        moon_dist = _angular_separation(self.ra, self.dec, conditions.moon_ra, conditions.moon_dec)
        if moon_dist > self.moon_limit:
            result = True
        return result


class CloseToTwilightBasisFunction(BaseBasisFunction):
    """Only execute if the sun is higher than `sun_alt_limit`,
    the current time is within `time_to_12deg` of -12 degree twilight,
    and there is at least `time_remaining` time left before -12 degree
    twilight.

    Is the current time and sun altitude close to twilight, but not too close?

    Parameters
    ----------
    max_sun_alt_limit : `float`
        The sun altitude limit (degrees). Sun must be higher than this
        at sunset to execute
    max_time_to_12deg : `float`
        How much time must be remaining before 12 degree twilight in
        the morning (minutes)
    min_time_remaining : `float`
        Minimum about of time that must be available before trying to
        execute (minutes)

    Notes
    -----
    This is primarily useful for surveys which must execute within close
    limits of -12 degree twilight.
    """

    def __init__(self, max_sun_alt_limit=-14.8, max_time_to_12deg=21.0, min_time_remaining=15.0):
        super().__init__()
        if max_time_to_12deg < min_time_remaining:
            raise ValueError(
                "time_to_12deg value of %f is less than time_remaining value of %f."
                % (max_time_to_12deg, min_time_remaining)
            )
        self.sun_alt_limit = np.radians(max_sun_alt_limit)
        self.time_to_12deg = max_time_to_12deg / 60.0 / 24.0
        self.min_time_remaining = min_time_remaining / 60.0 / 24.0

    def check_feasibility(self, conditions):
        result = False
        # If the sun is high, it's ok to execute
        if conditions.sun_alt > self.sun_alt_limit:
            result = True
        time_left = conditions.sun_n12_rising - conditions.mjd
        if time_left < self.time_to_12deg:
            result = True
        if time_left < self.min_time_remaining:
            result = False
        return result


class OnceInNightBasisFunction(BaseBasisFunction):
    """Stop observing if something has been executed already in the night

    Parameters
    ----------
    notes : `list` [ `str` ]
        A list of str to check if any observations with a matching note exist.
    """

    def __init__(self, notes=[]):
        super(OnceInNightBasisFunction, self).__init__()
        self.survey_features["note_in_night"] = features.NoteInNight(notes=notes)

    def check_feasibility(self, conditions):
        result = True
        if self.survey_features["note_in_night"].feature > 0:
            result = False
        return result


class SunAltHighLimitBasisFunction(BaseBasisFunction):
    """Don't observe unless the sun is above some limit

    Parameters
    ----------
    alt_limit : float (-15)
        Sun must be above alt_limit (degrees). Default -15.
    """

    def __init__(self, alt_limit=-15.0):
        super(SunAltHighLimitBasisFunction, self).__init__()
        self.alt_limit = np.radians(alt_limit)

    def check_feasibility(self, conditions):
        result = True
        if conditions.sun_alt < self.alt_limit:
            result = False
        return result


class NightModuloBasisFunction(BaseBasisFunction):
    """Only return true on certain nights"""

    def __init__(self, pattern=None):
        super(NightModuloBasisFunction, self).__init__()
        if pattern is None:
            pattern = [True, False]
        self.pattern = pattern
        self.mod_val = len(self.pattern)

    def check_feasibility(self, conditions):
        indx = int(conditions.night % self.mod_val)
        result = self.pattern[indx]
        return result


class TimeInTwilightBasisFunction(BaseBasisFunction):
    """Make sure there is some time left in twilight.

    Parameters
    ----------
    time_needed : float (5)
        The time needed remaining in twilight (minutes)
    """

    def __init__(self, time_needed=5.0):
        send_unused_deprecation_warning("TimeInTwilightBasisFunction")
        super(TimeInTwilightBasisFunction, self).__init__()
        self.time_needed = time_needed / 60.0 / 24.0  # To days

    def check_feasibility(self, conditions):
        result = False
        time1 = conditions.sun_n18_setting - conditions.mjd
        time2 = conditions.sun_n12_rising - conditions.mjd

        if time1 > self.time_needed:
            result = True
        else:
            if conditions.sun_alt > np.radians(-18.0):
                if time2 > self.time_needed:
                    result = True
        return result


class AfterEveningTwiBasisFunction(BaseBasisFunction):
    """Only execute right after evening twilight"""

    def __init__(self, time_after=30.0, alt_limit=18):
        super(AfterEveningTwiBasisFunction, self).__init__()
        self.time_after = IntRounded(time_after / 60.0 / 24.0)
        self.alt_limit = str(alt_limit)

    def check_feasibility(self, conditions):
        available_time = conditions.mjd - getattr(conditions, "sun_n" + self.alt_limit + "_setting")
        result = IntRounded(available_time) < self.time_after
        return result


class EndOfEveningBasisFunction(BaseBasisFunction):
    """Only let observations happen in a limited time before twilight"""

    def __init__(self, time_remaining=30.0, alt_limit=18):
        super(EndOfEveningBasisFunction, self).__init__()
        self.time_remaining = IntRounded(time_remaining / 60.0 / 24.0)
        self.alt_limit = str(alt_limit)

    def check_feasibility(self, conditions):
        available_time = getattr(conditions, "sun_n" + self.alt_limit + "_rising") - conditions.mjd
        result = IntRounded(available_time) < self.time_remaining
        return result


class TimeToTwilightBasisFunction(BaseBasisFunction):
    """Make sure there is enough time before twilight. Useful
    if you want to check before starting a long sequence of observations.

    Parameters
    ----------
    time_needed : float (30.)
        The time needed to run a survey (mintues).
    alt_limit : int (18)
        The sun altitude limit to use. Must be 12 or 18
    """

    def __init__(self, time_needed=30.0, alt_limit=18):
        super(TimeToTwilightBasisFunction, self).__init__()
        self.time_needed = time_needed / 60.0 / 24.0  # To days
        self.alt_limit = str(alt_limit)

    def check_feasibility(self, conditions):
        available_time = getattr(conditions, "sun_n" + self.alt_limit + "_rising") - conditions.mjd
        result = available_time > self.time_needed
        return result


class TimeToScheduledBasisFunction(BaseBasisFunction):
    """Make sure there is enough time before next scheduled observation.
    Useful if you want to check before starting a long sequence of
    observations.

    Parameters
    ----------
    time_needed : float (30.)
        The time needed to run a survey (mintues).
    """

    def __init__(self, time_needed=30.0):
        super(TimeToScheduledBasisFunction, self).__init__()
        self.time_needed = time_needed / 60.0 / 24.0  # To days

    def check_feasibility(self, conditions):
        if len(conditions.scheduled_observations) == 0:
            return True

        available_time = np.min(conditions.scheduled_observations) - conditions.mjd
        result = available_time > self.time_needed
        return result


class NotTwilightBasisFunction(BaseBasisFunction):
    def __init__(self, sun_alt_limit=-18):
        """
        # Should be -18 or -12
        """
        self.sun_alt_limit = str(int(sun_alt_limit)).replace("-", "n")
        super(NotTwilightBasisFunction, self).__init__()

    def check_feasibility(self, conditions):
        result = True
        if conditions.mjd < getattr(conditions, "sun_" + self.sun_alt_limit + "_setting"):
            result = False
        if conditions.mjd > getattr(conditions, "sun_" + self.sun_alt_limit + "_rising"):
            result = False
        return result


class ForceDelayBasisFunction(BaseBasisFunction):
    """Keep a survey from executing too rapidly.

    Parameters
    ----------
    days_delay : `float`, optional
        The number of days to force a gap on.
    scheduler_note : `str` or None, optional
        The value of the scheduler_note to count.
        Default None will not consider scheduler_note.
    survey_name : `str` or None, optional
        Backwards compatible version of scheduler_note.
    """

    def __init__(self, days_delay=2.0, scheduler_note=None, survey_name=None):
        super(ForceDelayBasisFunction, self).__init__()
        self.days_delay = days_delay
        if scheduler_note is None and survey_name is not None:
            warnings.warn("survey_name deprecated in favor of scheduler_note", FutureWarning)
            self.scheduler_note = survey_name
        else:
            self.scheduler_note = scheduler_note
        self.survey_features["last_obs_self"] = features.LastObservation(scheduler_note=self.scheduler_note)

    def check_feasibility(self, conditions):
        result = True
        if conditions.mjd - self.survey_features["last_obs_self"].feature["mjd"] < self.days_delay:
            result = False
        return result


class SoftDelayBasisFunction(BaseBasisFunction):
    """Like Force_delay, but go ahead and let things catch up if they fall
    far behind.

    Parameters
    ----------
    fractions : `list` [`float`]
    delays : `list` [`float`]
    scheduler_note : `str` or None, optional
        The scheduler_note to identify observations from a given survey or
        survey mode.
    survey_name : `str` or None, optional
        Deprecated version of scheduler_note. Overriden by scheduler_note
        if not None.
    """

    def __init__(
        self, fractions=[0.000, 0.009, 0.017], delays=[0.0, 0.5, 1.5], scheduler_note=None, survey_name=None
    ):
        if len(fractions) != len(delays):
            raise ValueError("fractions and delays must be same length")
        super(SoftDelayBasisFunction, self).__init__()
        self.delays = delays
        if scheduler_note is None and survey_name is not None:
            warnings.warn("survey_name deprecated in favor of scheduler_note", FutureWarning)
            self.scheduler_note = survey_name
        else:
            self.scheduler_note = scheduler_note
        self.survey_features["last_obs_self"] = features.LastObservation(scheduler_note=self.scheduler_note)
        self.fractions = fractions
        self.survey_features["N_total"] = features.NObsCount(note=None)
        self.survey_features["N_note"] = features.NObsCount(note=self.scheduler_note)

    def check_feasibility(self, conditions):
        result = True
        current_ratio = self.survey_features["N_note"].feature / self.survey_features["N_total"].feature
        indx = np.searchsorted(self.fractions, current_ratio)
        if indx == len(self.fractions):
            indx -= 1
        delay = self.delays[indx]
        if conditions.mjd - self.survey_features["last_obs_self"].feature["mjd"] < delay:
            result = False
        return result


class HourAngleLimitBasisFunction(BaseBasisFunction):
    """Only execute a survey in limited hour angle ranges. Useful for
    limiting Deep Drilling Fields.

    Parameters
    ----------
    RA : float (0.)
        RA of the target (degrees).
    ha_limits : list of lists
        limits for what hour angles are acceptable (hours). e.g.,
        to give 4 hour window around RA=0, ha_limits=[[22,24], [0,2]]
    """

    def __init__(self, RA=0.0, ha_limits=[[22, 24], [0, 2]]):
        super(HourAngleLimitBasisFunction, self).__init__()
        self.ra_hours = RA / 360.0 * 24.0
        self.ha_limits = np.array(ha_limits)

    def check_feasibility(self, conditions):
        target_ha = (conditions.lmst - self.ra_hours) % 24
        # Are we in any of the possible windows
        result = False
        for limit in self.ha_limits:
            lres = limit[0] <= target_ha < limit[1]
            result = result or lres

        return result


class MoonDownBasisFunction(BaseBasisFunction):
    """Demand the moon is down"""

    def check_feasibility(self, conditions):
        result = True
        if conditions.moon_alt > 0:
            result = False
        return result


class FractionOfObsBasisFunction(BaseBasisFunction):
    """Limit the fraction of all observations that can be labelled a certain
    scheduler note.

    Useful for keeping DDFs from exceeding a given fraction
    of the total survey.

    Parameters
    ----------
    frac_total : `float`
        The fraction of total observations that can be of this survey
    scheduler_note : `str` or None, optional
        The scheduler_note to identify observations from a given survey or
        survey mode.
    survey_name : `str` or None, optional
        Deprecated version of scheduler_note. Overriden by scheduler_note
        if scheduler_note not None.
    """

    def __init__(self, frac_total, scheduler_note=None, survey_name=None):
        super(FractionOfObsBasisFunction, self).__init__()
        if scheduler_note is None and survey_name is not None:
            warnings.warn("survey_name deprecated in favor of scheduler_note", FutureWarning)
            self.scheduler_note = survey_name
        else:
            self.scheduler_note = scheduler_note
        self.frac_total = frac_total
        self.survey_features["N_total"] = features.NObsCount(note=None)
        self.survey_features["N_note"] = features.NObsCount(note=self.scheduler_note)

    def check_feasibility(self, conditions):
        # If nothing has been observed, fine to go
        result = True
        if self.survey_features["N_total"].feature > 0:
            ratio = self.survey_features["N_note"].feature / self.survey_features["N_total"].feature
            if ratio > self.frac_total:
                result = False
        return result


class LookAheadDdfBasisFunction(BaseBasisFunction):
    """Look into the future to decide if it's a good time to observe or block.

    Parameters
    ----------
    frac_total : float
        The fraction of total observations that can be of this survey
    aggressive_fraction : float
        If the fraction of observations drops below ths value, be more
        aggressive in scheduling.  e.g., do not wait for conditions to
        improve, execute as soon as possible.
    time_needed : float (30.)
        Estimate of the amount of time needed to execute DDF sequence
        (minutes).
    RA : float (0.)
        The RA of the DDF
    ha_limits : list of lists (None)
        limits for what hour angles are acceptable (hours). e.g.,
        to give 4 hour window around HA=0, ha_limits=[[22,24], [0,2]]
    scheduler_note : `str` or None, optional
        The scheduler_note to identify observations from a given survey or
        survey mode.
    survey_name : `str` or None, optional
        Deprecated version of scheduler_note. Overriden by scheduler_note
        if scheduler_note not None.
    time_jump : float (44.)
        The amount of time to assume will jump ahead if another survey
        executes (minutes)
    sun_alt_limit : float (-18.)
        The limit to assume twilight starts (degrees)
    """

    def __init__(
        self,
        frac_total,
        aggressive_fraction,
        time_needed=30.0,
        RA=0.0,
        ha_limits=None,
        scheduler_note=None,
        survey_name=None,
        time_jump=44.0,
        sun_alt_limit=-18.0,
    ):
        super(LookAheadDdfBasisFunction, self).__init__()
        if aggressive_fraction > frac_total:
            raise ValueError("aggressive_fraction should be less than frac_total")
        if scheduler_note is None and survey_name is not None:
            warnings.warn("survey_name deprecated in favor of scheduler_note", FutureWarning)
            self.scheduler_note = survey_name
        else:
            self.scheduler_note = scheduler_note
        self.frac_total = frac_total
        self.ra_hours = RA / 360.0 * 24.0
        self.ha_limits = np.array(ha_limits)
        self.sun_alt_limit = str(int(sun_alt_limit)).replace("-", "n")
        self.time_jump = time_jump / 60.0 / 24.0  # To days
        self.time_needed = time_needed / 60.0 / 24.0  # To days
        self.aggressive_fraction = aggressive_fraction
        self.survey_features["N_total"] = features.NObsCount(note=None)
        self.survey_features["N_note"] = features.NObsCount(note=self.scheduler_note)

    def check_feasibility(self, conditions):
        result = True
        target_ha = (conditions.lmst - self.ra_hours) % 24
        ratio = self.survey_features["N_note"].feature / self.survey_features["N_total"].feature
        available_time = getattr(conditions, "sun_" + self.sun_alt_limit + "_rising") - conditions.mjd
        # If it's more that self.time_jump to hour angle zero
        # See if there will be enough time to twilight in the future
        if (IntRounded(target_ha) > IntRounded(12)) & (
            IntRounded(target_ha) < IntRounded(24.0 - self.time_jump)
        ):
            if IntRounded(available_time) > IntRounded(self.time_needed + self.time_jump):
                result = False
                # If we paused for better conditions, but the moon will
                # rise, turn things back on.
                if IntRounded(conditions.moon_alt) < IntRounded(0):
                    if IntRounded(conditions.moonrise) > IntRounded(conditions.mjd):
                        if IntRounded(conditions.moonrise - conditions.mjd) > IntRounded(self.time_jump):
                            result = True
        # If the moon is up and will set soon, pause
        if IntRounded(conditions.moon_alt) > IntRounded(0):
            time_after_moonset = (
                getattr(conditions, "sun_" + self.sun_alt_limit + "_rising") - conditions.moonset
            )
            if IntRounded(conditions.moonset) > IntRounded(self.time_jump):
                if IntRounded(time_after_moonset) > IntRounded(self.time_needed):
                    result = False

        # If the survey has fallen far behind, be agressive and observe
        # anytime it's up.
        if IntRounded(ratio) < IntRounded(self.aggressive_fraction):
            result = True
        return result


class CloudedOutBasisFunction(BaseBasisFunction):
    def __init__(self, cloud_limit=0.7):
        super(CloudedOutBasisFunction, self).__init__()
        self.cloud_limit = cloud_limit

    def check_feasibility(self, conditions):
        result = True
        if conditions.bulk_cloud > self.cloud_limit:
            result = False
        return result


class RisingMoreBasisFunction(BaseBasisFunction):
    """Say a spot is not available if it will rise substantially before
    twilight.

    Parameters
    ----------
    RA : float
        The RA of the point in the sky (degrees)
    pad : float
        When to start observations if there's plenty of time before
        twilight (minutes)
    """

    def __init__(self, RA, pad=30.0):
        super(RisingMoreBasisFunction, self).__init__()
        self.ra_hours = RA * 24 / 360.0
        self.pad = pad / 60.0  # To hours

    def check_feasibility(self, conditions):
        result = True
        # Calculating hour angle instead of using conditions.HA
        # because want -12 to +12 hours instead of 0 - 2pi values.
        hour_angle = conditions.lmst - self.ra_hours
        # If it's rising, and twilight is well beyond when it crosses
        # the meridian
        time_to_twi = (conditions.sun_n18_rising - conditions.mjd) * 24.0
        if (hour_angle < -self.pad) & (np.abs(hour_angle) < (time_to_twi - self.pad)):
            result = False
        return result


class SunAltLimitBasisFunction(BaseBasisFunction):
    """Only observe if the sun is below a given altitude limit.

    Parameters
    ----------
    alt_limit : `float`
        The maximum altitude for the sun.
    """

    def __init__(self, alt_limit=-12.1):
        super(SunAltLimitBasisFunction, self).__init__()
        self.alt_limit = np.radians(alt_limit)

    def check_feasibility(self, conditions):
        result = True
        if conditions.sun_alt > self.alt_limit:
            result = False
        return result
