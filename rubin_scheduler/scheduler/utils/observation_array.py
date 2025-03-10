__all__ = (
    "ObservationArray",
    "ScheduledObservationArray",
    "backfill_filter_from_band",
)

import numpy as np

HANDLED_FUNCTIONS = {}


def backfill_filter_from_band(observation_array):
    """Simple utility to backfill the `band` values in an ObservationArray
    if not otherwise present.

    Parameters
    ----------
    observation_array : ObservationArray
        The observation array to ensure that 'band' is present in, if
        only physical filtername was present.

    Returns
    -------
    observation_array : ObservationArray
        The same observation_array, but with `band` added where it was missing.

    Notes
    -----
    This is primarily a utility to provide backwards compatibility for
    scheduler configurations where ObservationArrays are input (such as
    for FieldSurvey), but only have `filter` present, instead of also `band`.
    For the current setup for comcam and lsstcam, the stripping of
    "_0123456789" is sufficient, in case the configuration used physical
    filtername --
    physical filtername <> band includes "r_03" <> "r", for example.
    """
    missing_band = np.where(observation_array["band"] == "")
    band = np.char.rstrip(observation_array["filter"][missing_band], chars="_0123456789")
    observation_array["band"][missing_band] = band
    return observation_array


class ObservationArray(np.ndarray):
    """Class to work as an array of observations

    Parameters
    ----------
    n : `int`
        Size of array to return. Default 1.

    The numpy fields have the following labels.

    RA : `float`
       The Right Acension of the observation (center of the field)
       (Radians)
    dec : `float`
       Declination of the observation (Radians)
    mjd : `float`
       Modified Julian Date at the start of the observation
       (time shutter opens)
    exptime : `float`
       Total exposure time of the visit (seconds)
    band : `str`
        The band used. Should be one of u, g, r, i, z, y.
    filter : `str`
        The physical filter name.
    rotSkyPos : `float`
        The rotation angle of the camera relative to the sky E of N
        (Radians). Will be ignored if rotTelPos is finite.
        If rotSkyPos is set to NaN, rotSkyPos_desired is used.
    rotTelPos : `float`
        The rotation angle of the camera relative to the telescope
        (radians). Set to np.nan to force rotSkyPos to be used.
    rotSkyPos_desired : `float`
        If both rotSkyPos and rotTelPos are None/NaN, then
        rotSkyPos_desired (radians) is used. If rotSkyPos_desired
        results in a valid rotTelPos, rotSkyPos is set to
        rotSkyPos_desired. If rotSkyPos and rotTelPos are both NaN,
        and rotSkyPos_desired results in an out of range value for the
        camera rotator, then rotTelPos_backup is used.
    rotTelPos_backup : `float`
        Rotation angle of the camera relative to the telescope (radians).
        Only used as a last resort if rotSkyPos and rotTelPos are set
        to NaN and rotSkyPos_desired results in an out of range rotator
        value.
    nexp : `int`
        Number of exposures in the visit.
    flush_by_mjd : `float`
        If we hit this MJD, we should flush the queue and refill it.
    scheduler_note : `str` (optional)
        Usually good to set the note field so one knows which survey
        object generated the observation.
    target_name : `str` (optional)
        A note about what target is being observed.
        This maps to target_name in the ConsDB.
        Generally would be used to identify DD, ToO or special targets.
    science_program : `str` (optional)
        Science program being executed.
        This maps to science_program in the ConsDB, although can
        be overwritten in JSON BLOCK.
        Generally would be used to identify a particular program for DM.
    observation_reason : `str` (optional)
        General 'reason' for observation, for DM purposes.
        (for scheduler purposes, use `scheduler_note`).
        This maps to observation_reason in the ConsDB, although could
        be overwritten in JSON BLOCK.
        Most likely this is just "science" or "FBS" when using the FBS.
    cloud_extinction : float
        Cloud extinction that was applied by the model observatory (mags).

    Notes
    -----

    On the camera rotator angle. Order of priority goes:
    rotTelPos > rotSkyPos > rotSkyPos_desired > rotTelPos_backup
    where if rotTelPos is NaN, it checks rotSkyPos. If rotSkyPos is set,
    but not at an accessible rotTelPos, the observation will fail.
    If rotSkyPos is NaN, then rotSkyPos_desired is used. If
    rotSkyPos_desired is at an inaccessbile rotTelPos, the observation
    does not fail, but falls back to the value in rotTelPos_backup.

    Lots of additional fields that get filled in by the model observatory
    when the observation is completed.
    See documentation at:
    https://rubin-scheduler.lsst.io/output_schema.html

    """

    def __new__(cls, n=1):
        dtypes = [
            ("ID", int),
            ("RA", float),
            ("dec", float),
            ("mjd", float),
            ("flush_by_mjd", float),
            ("exptime", float),
            ("band", "U40"),
            ("filter", "U40"),
            ("rotSkyPos", float),
            ("rotSkyPos_desired", float),
            ("nexp", int),
            ("airmass", float),
            ("FWHM_500", float),
            ("FWHMeff", float),
            ("FWHM_geometric", float),
            ("skybrightness", float),
            ("night", int),
            ("slewtime", float),
            ("visittime", float),
            ("slewdist", float),
            ("fivesigmadepth", float),
            ("alt", float),
            ("az", float),
            ("pa", float),
            ("pseudo_pa", float),
            ("clouds", float),
            ("moonAlt", float),
            ("sunAlt", float),
            ("scheduler_note", "U40"),
            ("target_name", "U40"),
            ("target_id", int),
            ("lmst", float),
            ("rotTelPos", float),
            ("rotTelPos_backup", float),
            ("moonAz", float),
            ("sunAz", float),
            ("sunRA", float),
            ("sunDec", float),
            ("moonRA", float),
            ("moonDec", float),
            ("moonDist", float),
            ("solarElong", float),
            ("moonPhase", float),
            ("cummTelAz", float),
            ("observation_reason", "U40"),
            ("science_program", "U40"),
            ("cloud_extinction", float),
            # TODO : Remove this hack which is for use with ts_scheduler
            # version <=v2.3 .. remove ts_scheduler actually drops "note".
            ("note", "U40"),
        ]
        obj = np.zeros(n, dtype=dtypes).view(cls)
        return obj

    def tolist(self):
        """Convert to a list of 1-element arrays"""
        obs_list = []
        for obs in self:
            new_obs = self.__class__(n=1)
            new_obs[0] = obs
            obs_list.append(new_obs)

        return obs_list

    def __array_function__(self, func, types, args, kwargs):
        # If we want "standard numpy behavior",
        # convert any ObservationArray to ndarray views
        if func not in HANDLED_FUNCTIONS:
            new_args = []
            for arg in args:
                if issubclass(arg.__class__, ObservationArray):
                    new_args.append(arg.view(np.ndarray))
                else:
                    new_args.append(arg)
            return func(*new_args, **kwargs)
        if not all(issubclass(t, ObservationArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(numpy_function):
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate(arrays):
    result = arrays[0].__class__(n=sum(len(a) for a in arrays))
    return np.concatenate([np.asarray(a) for a in arrays], out=result)


@implements(np.in1d)
def in1d(ar1, ar2, **kwargs):
    # Just here to get rid of DeprecationWarning. Can probably
    # be removed after a future numpy update.
    return np.isin(ar1, ar2, **kwargs)


class ScheduledObservationArray(ObservationArray):
    """Make an array to hold pre-scheduling observations

    Note
    ----
    mjd_tol : `float`
        The tolerance on how early an observation can execute (days).
        Observation will be considered valid to attempt
        when mjd-mjd_tol < current MJD < flush_by_mjd (and other
        conditions below pass)
    dist_tol : `float`
        The angular distance an observation can be away from the
        specified RA,Dec and still count as completing the observation
        (radians).
    alt_min : `float`
        The minimum altitude to consider executing the observation
        (radians).
    alt_max : `float`
        The maximuim altitude to try observing (radians).
    HA_max : `float`
        Hour angle limit. Constraint is such that for hour angle
        running from 0 to 24 hours, the target RA,Dec must be greather
        than HA_max and less than HA_min. Set HA_max to 0 for no
        limit. (hours)
    HA_min : `float`
        Hour angle limit. Constraint is such that for hour angle
        running from 0 to 24 hours, the target RA,Dec must be greather
        than HA_max and less than HA_min. Set HA_min to 24 for
        no limit. (hours)
    sun_alt_max : `float`
        The sun must be below sun_alt_max to execute. (radians)
    moon_min_distance : `float`
        The minimum distance to demand the moon should be away (radians)
    observed : `bool`
        If set to True, scheduler will probably consider this a
        completed observation and never attempt it.

    """

    def __new__(cls, n=1):
        # Standard things from the usual observations
        dtypes1 = [
            ("ID", int),
            ("RA", float),
            ("dec", float),
            ("mjd", float),
            ("flush_by_mjd", float),
            ("exptime", float),
            ("band", "U1"),
            ("filter", "U40"),
            ("rotSkyPos", float),
            ("rotTelPos", float),
            ("rotTelPos_backup", float),
            ("rotSkyPos_desired", float),
            ("nexp", int),
            ("scheduler_note", "U40"),
            ("target_name", "U40"),
            ("science_program", "U40"),
            ("observation_reason", "U40"),
        ]

        # New things not in standard ObservationArray
        dtype2 = [
            ("mjd_tol", float),
            ("dist_tol", float),
            ("alt_min", float),
            ("alt_max", float),
            ("HA_max", float),
            ("HA_min", float),
            ("sun_alt_max", float),
            ("moon_min_distance", float),
            ("observed", bool),
        ]

        obj = np.zeros(n, dtype=dtypes1 + dtype2).view(cls)
        return obj

    def to_observation_array(self):
        """Convert the scheduled observation to a
        Regular ObservationArray
        """
        result = ObservationArray(n=self.size)
        in_common = np.intersect1d(self.dtype.names, result.dtype.names)
        for key in in_common:
            result[key] = self[key]
        return result
