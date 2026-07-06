import numpy as np

from rubin_scheduler.utils import _approx_ra_dec2_alt_az

__all__ = [
    "TargetoO",
    "SimTargetooServer",
]


class TargetoO:
    """Class to hold information about a target of opportunity object

    Parameters
    ----------
    tooid : `str`
        Unique ID for the ToO. Probably using `source` from EFD.
    footprints : `np.array`
        np.array healpix maps. 1 for areas to observe, 0 for no observe.
        Can use np.nan for no-observe pixels, but that will be interpreted
        to mean the map cannot expand if the resolution chages.
    mjd_start : `float`
        The MJD the ToO starts
    duration : `float`
        Duration of the ToO (days).
    ra_rad_center : `float`
        RA of the estimated center of the event (radians).
    dec_rad_center : `float`
        Dec of the estimated center of the event (radians).
    too_type : `str`
        The type of ToO that is made.
    posterior_distance : `float`
        The posterior distance of the event. (kpc)
    interrupt_queue : `bool`
        This ToO is urgent, so if it is high enoug in the
        sky, the scheduler should flush its queue so ToO
        observations can start without waiting.
    alt_limit : `float`
        Altitude limit that some part of the ToO footprint
        must be above to trigger a queue flush (degrees).
    """

    def __init__(
        self,
        tooid,
        footprint,
        mjd_start,
        duration=None,
        ra_rad_center=None,
        dec_rad_center=None,
        too_type=None,
        posterior_distance=None,
        interrupt_queue=True,
        alt_limit=20,
    ):
        self.footprint = footprint
        self.duration = duration
        self.id = tooid
        self.mjd_start = mjd_start
        self.ra_rad_center = ra_rad_center
        self.dec_rad_center = dec_rad_center
        self.too_type = too_type
        self.posterior_distance = posterior_distance

        self.alt_limit = np.radians(alt_limit)
        self.interrupt_queue = interrupt_queue

    def queue_should_flush(self, conditions):
        """Given current conditions, is the ToO
        probably visible and should interrupt the queue
        """

        # Kwarg set saying don't interrupt
        if not self.interrupt_queue:
            return False

        result = True

        # If we have a ra,dec center, check it is above alt limit
        if self.ra_rad_center is not None:
            alt, az = _approx_ra_dec2_alt_az(
                self.ra_rad_center,
                self.dec_rad_center,
                conditions.site.latitude_rad,
                conditions.site.longitude_rad,
                conditions.mjd,
            )
            if alt < self.alt_limit:
                result = False
        # No ra,dec center, check if any part of footprint
        # is above alt limit.
        else:
            indx = np.where(conditions.alt >= self.alt_limit)[0]
            # footprint pixels could be NaN for not-observe, use 0 here.
            fp_pix = np.nan_to_num(self.footprint[indx], copy=True, nan=0)
            if np.max(fp_pix) == 0:
                result = False

        return result


class SimTargetooServer:
    """Wrapper to deliver a targetoO object at the right time"""

    def __init__(self, targeto_o_list):
        self.targeto_o_list = targeto_o_list
        self.mjd_starts = np.array([too.mjd_start for too in self.targeto_o_list])
        durations = np.array([too.duration for too in self.targeto_o_list], dtype="float")
        # Fill any Nans with a default value.
        # This should never be necessary in full simulations, where duration
        # is always set by gen_events.
        np.nan_to_num(durations, copy=False, nan=3)
        self.mjd_ends = self.mjd_starts + durations

    def __call__(self, mjd):
        in_range = np.where((mjd > self.mjd_starts) & (mjd < self.mjd_ends))[0]
        result = None
        if in_range.size > 0:
            result = [self.targeto_o_list[i] for i in in_range]
        return result
