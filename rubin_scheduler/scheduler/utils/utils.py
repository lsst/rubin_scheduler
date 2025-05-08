__all__ = (
    "IntRounded",
    "int_binned_stat",
    "smallest_signed_angle",
    "SchemaConverter",
    "HpInComcamFov",
    "HpInLsstFov",
    "hp_kd_tree",
    "match_hp_resolution",
    "TargetoO",
    "SimTargetooServer",
    "restore_scheduler",
    "warm_start",
    "gnomonic_project_toxy",
    "gnomonic_project_tosky",
    "raster_sort",
    "run_info_table",
    "inrange",
    "season_calc",
    "create_season_offset",
    "thetaphi2xyz",
    "xyz2thetaphi",
    "mean_azimuth",
    "wrap_ra_dec",
    "rotx",
)

import copy
import datetime
import os
import socket
import sqlite3
import warnings

import healpy as hp
import matplotlib.path as mplPath
import numpy as np
import pandas as pd

from rubin_scheduler import __version__
from rubin_scheduler.scheduler.utils.observation_array import ObservationArray
from rubin_scheduler.utils import (
    DEFAULT_NSIDE,
    _build_tree,
    _hpid2_ra_dec,
    _xyz_from_ra_dec,
    xyz_angular_radius,
)

rsVersion = __version__


def smallest_signed_angle(a1, a2):
    """
    via https://stackoverflow.com/questions/1878907/
    the-smallest-difference-between-2-angles
    """
    two_pi = 2.0 * np.pi
    x = a1 % two_pi
    y = a2 % two_pi
    a = (x - y) % two_pi
    b = (y - x) % two_pi
    result = b + 0
    alb = np.where(a < b)[0]
    result[alb] = -1.0 * a[alb]
    return result


def thetaphi2xyz(theta, phi):
    """Convert theta,phi to x,y,z position on the unit sphere

    Parameters
    ----------
    theta : `float`
        Theta coordinate in radians (should be 0-2pi).
    phi : `float`
        Phi coordinate in radians (should run from 0-pi).
    """
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z


def xyz2thetaphi(x, y, z):
    """x,y,z position on unit sphere to theta,phi coords."""
    phi = np.arccos(z)
    theta = np.arctan2(y, x)
    return theta, phi


def rotx(theta, x, y, z):
    """rotate the x,y,z points theta radians about x axis"""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xp = x
    yp = y * cos_t + z * sin_t
    zp = -y * sin_t + z * cos_t
    return xp, yp, zp


def wrap_ra_dec(ra, dec):
    # XXX--from MAF, should put in general utils
    """
    Wrap RA into 0-2pi and Dec into +/0 pi/2.

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians
    dec : numpy.ndarray
        Dec in radians

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Wrapped RA/Dec values, in radians.
    """
    # Wrap dec.
    low = np.where(dec < -np.pi / 2.0)[0]
    dec[low] = -1 * (np.pi + dec[low])
    ra[low] = ra[low] - np.pi
    high = np.where(dec > np.pi / 2.0)[0]
    dec[high] = np.pi - dec[high]
    ra[high] = ra[high] - np.pi
    # Wrap RA.
    ra = ra % (2.0 * np.pi)
    return ra, dec


def mean_azimuth(az, min_val=0.1):
    """Compute the mean azimuth value accounting for wrap

    Parameters
    ----------
    az : `array-like`
        The azimuths to average. Radians
    min_val : `float`
        A min cutoff to just use pi as the mean. Default 0.1. Radians
    """

    x = np.cos(az)
    y = np.sin(az)
    meanx = np.mean(x)
    meany = np.mean(y)
    angle = np.arctan2(meany, meanx)
    radius = np.sqrt(meanx**2 + meany**2)
    mid_az = angle % (2.0 * np.pi)
    if IntRounded(radius) < IntRounded(min_val):
        mid_az = np.pi
    return mid_az


class IntRounded:
    """
    Class to help force comparisons be made on scaled up integers,
    preventing machine precision issues cross-platforms.

    Danger zone would be doing comparisons with differences between
    `scale` and `limit`, e.g.:
    (1 + 1e-9) == (1 + 1e-10)
    returns False as expected, but
    IntRounded(1 + 1e-9) == intRounded(1 + 1e-10)
    will return True because it rounds off at 9 decimal places.

    Note that casting a NaN to an int can have different behaviours
    cross-platforms, so will throw an error if attempted.

    Parameters
    ----------
    inval : number-like thing
        Some number/array that we will want to compare
    scale : `float`
        How much to scale inval before rounding and converting to an int.
        Essentially how many digits of precesion to use in comparisons.
        Default 1e6.
    limit : `float`
        If values have a differce less than the limit, force comparisions
        to be done with scaled ints. Should be larger than whatever you
        think the machine floating point precision might be.
        Default 1e-9.
    """

    def __init__(self, inval, scale=1e6, limit=1e-9):
        if np.any(~np.isfinite(inval)):
            raise ValueError("IntRounded can only take finite values.")
        self.initial = inval
        self.scale = scale
        self.limit = limit

    def check_limit(self, v1, v2):
        """See if two values are potentiall close to floating precision limit.
        If they are close, return scaled up integers
        """
        v1 = np.atleast_1d(copy.copy(v1))
        v2 = np.atleast_1d(copy.copy(v2))

        non_zero_v1 = np.where(v1 != 0)
        non_zero_v2 = np.where(v2 != 0)

        # Don't need to worry if comparing arrays
        # that are empty or only zeros
        if (len(v1[non_zero_v1]) == 0) | (len(v2[non_zero_v2]) == 0):
            return v1, v2

        # Scale factor so we correctly compare things like 1e-25 to 1e-26
        # Kinda messy to avoid zeros or negatives to log.
        order_of_mag = np.min(
            [
                np.floor(np.log10(np.abs(v1[non_zero_v1]))).min(),
                np.floor(np.log10(np.abs(v2[non_zero_v2]))).min(),
            ]
        )
        scale_factor = 10**order_of_mag

        ab_diff = np.abs(v1 / scale_factor - v2 / scale_factor)
        close = np.where((ab_diff <= self.limit))[0]
        # There are some values with a potential round error
        if close.size > 0:
            floor = np.floor(np.minimum(v1, v2))
            v1 = np.round(v1 - floor * self.scale).astype(int)
            v2 = np.round(v2 - floor * self.scale).astype(int)
        return v1, v2

    def __eq__(self, other):
        v1, v2 = self.check_limit(self.initial, other.initial)
        return v1 == v2

    def __ne__(self, other):
        v1, v2 = self.check_limit(self.initial, other.initial)
        return v1 != v2

    def __lt__(self, other):
        v1, v2 = self.check_limit(self.initial, other.initial)
        return v1 < v2

    def __le__(self, other):
        v1, v2 = self.check_limit(self.initial, other.initial)
        return v1 <= v2

    def __gt__(self, other):
        v1, v2 = self.check_limit(self.initial, other.initial)
        return v1 > v2

    def __ge__(self, other):
        v1, v2 = self.check_limit(self.initial, other.initial)
        return v1 >= v2

    def __repr__(self):
        return str(self.initial)

    def __add__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = IntRounded(self.initial + other.initial, scale=out_scale)
        return result

    def __sub__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = IntRounded(self.initial - other.initial, scale=out_scale)
        return result

    def __mul__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = IntRounded(self.initial * other.initial, scale=out_scale)
        return result

    def __div__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = IntRounded(self.initial / other.initial, scale=out_scale)
        return result


def restore_scheduler(observation_id, scheduler, observatory, in_obs, band_sched=None, fast=True):
    """Put the scheduler and observatory in the state they were in.
    Handy for checking reward fucnction

    Parameters
    ----------
    observation_id : int
        The ID of the last observation that should be completed
    scheduler : rubin_scheduler.scheduler.scheduler object
        Scheduler object.
    observatory : rubin_scheduler.scheduler.observatory.Model_observatory
        The observaotry object
    in_obs : np.array or str
        Array of observations (formated like
        rubin_scheduler.scheduler.ObservationArray). If a string,
        assumed to be a file and SchemaConverter is used to load it.
    band_sched : rubin_scheduler.scheduler.scheduler object
        The band scheduler. Note that we don't look up the official
        end of the previous night, so there is potential for the
        loaded bands to not match.
    fast : bool (True)
        If True, loads observations and passes them as an array to
        the `add_observations_array` method. If False,
        passes observations individually with `add_observation` method.
    """
    if isinstance(in_obs, str):
        sc = SchemaConverter()
        # load up the observations
        observations = sc.opsim2obs(in_obs)
    else:
        observations = in_obs
    good_obs = np.where(observations["ID"] <= observation_id)[0]
    observations = observations[good_obs]

    if fast:
        scheduler.add_observations_array(observations)
        obs = observations[-1]
    else:
        for obs in observations:
            scheduler.add_observation(obs)

    if band_sched is not None:
        # We've assumed the band scheduler doesn't have any bands
        # May need to call the add_observation method on it if that
        # changes.

        # Make sure we have mounted the right bands for the night
        # XXX--note, this might not be exact, but should work most
        # of the time.
        mjd_start_night = np.min(observations["mjd"][np.where(observations["night"] == obs["night"])])
        observatory.mjd = mjd_start_night
        conditions = observatory.return_conditions()
        bands_needed = band_sched(conditions)
    else:
        bands_needed = ["u", "g", "r", "i", "y"]

    # update the observatory
    observatory.mjd = obs["mjd"] + observatory.observatory.visit_time(obs) / 3600.0 / 24.0
    observatory.obs_id_counter = obs["ID"] + 1
    observatory.observatory.parked = False
    observatory.observatory.current_ra_rad = obs["RA"]
    observatory.observatory.current_dec_rad = obs["dec"]
    observatory.observatory.current_rot_sky_pos_rad = obs["rotSkyPos"]
    observatory.observatory.cumulative_azimuth_rad = obs["cummTelAz"]
    observatory.observatory.current_band = obs["band"]
    observatory.observatory.mounted_bands = bands_needed
    # Note that we haven't updated last_az_rad, etc, but those
    # values should be ignored.

    return scheduler, observatory


def int_binned_stat(ids, values, statistic=np.mean):
    """
    Like scipy.binned_statistic, but for unique int ids
    """

    uids = np.unique(ids)
    order = np.argsort(ids)

    ordered_ids = ids[order]
    ordered_values = values[order]

    left = np.searchsorted(ordered_ids, uids, side="left")
    right = np.searchsorted(ordered_ids, uids, side="right")

    stat_results = []
    for le, ri in zip(left, right):
        stat_results.append(statistic(ordered_values[le:ri]))

    return uids, np.array(stat_results)


def gnomonic_project_toxy(ra1, dec1, r_acen, deccen):
    """Calculate x/y projection of ra1/dec1 in system with center
    at r_acen, deccen. Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(deccen) * np.sin(dec1) + np.cos(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)
    x = np.cos(dec1) * np.sin(ra1 - r_acen) / cosc
    y = (np.cos(deccen) * np.sin(dec1) - np.sin(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, r_acen, deccen):
    """Calculate RA/dec on sky of object with x/y and RA/Cen of
    field of view. Returns Ra/dec in radians."""
    denom = np.cos(deccen) - y * np.sin(deccen)
    RA = r_acen + np.arctan2(x, denom)
    dec = np.arctan2(np.sin(deccen) + y * np.cos(deccen), np.sqrt(x * x + denom * denom))
    return RA, dec


def match_hp_resolution(in_map, nside_out, unseen2nan=True):
    """Utility to convert healpix map resolution if needed and
    change hp.UNSEEN values to np.nan.

    Parameters
    ----------
    in_map : np.array
        A valie healpix map
    nside_out : int
        The desired resolution to convert in_map to
    unseen2nan : bool (True)
        If True, convert any hp.UNSEEN values to np.nan
    """
    current_nside = hp.npix2nside(np.size(in_map))
    if current_nside != nside_out:
        out_map = hp.ud_grade(in_map, nside_out=nside_out)
    else:
        out_map = in_map
    if unseen2nan:
        out_map[np.where(out_map == hp.UNSEEN)] = np.nan
    return out_map


def raster_sort(x0, order=["x", "y"], xbin=1.0):
    """XXXX--depriciated, use tsp instead.

    Do a sort to scan a grid up and down. Simple starting guess
    to traveling salesman.

    Parameters
    ----------
    x0 : array
    order : list
        Keys for the order x0 should be sorted in.
    xbin : float (1.)
        The bin_size to round off the first coordinate into

    returns
    -------
    array sorted so that it rasters up and down.
    """
    coords = x0.copy()
    bins = np.arange(
        coords[order[0]].min() - xbin / 2.0,
        coords[order[0]].max() + 3.0 * xbin / 2.0,
        xbin,
    )
    # digitize my bins
    coords[order[0]] = np.digitize(coords[order[0]], bins)
    order1 = np.argsort(coords, order=order)
    coords = coords[order1]
    places_to_invert = np.where(np.diff(coords[order[-1]]) < 0)[0]
    if np.size(places_to_invert) > 0:
        places_to_invert += 1
        indx = np.arange(coords.size)
        index_sorted = np.zeros(indx.size, dtype=int)
        index_sorted[0 : places_to_invert[0]] = indx[0 : places_to_invert[0]]

        for i, inv_pt in enumerate(places_to_invert[:-1]):
            if i % 2 == 0:
                index_sorted[inv_pt : places_to_invert[i + 1]] = indx[inv_pt : places_to_invert[i + 1]][::-1]
            else:
                index_sorted[inv_pt : places_to_invert[i + 1]] = indx[inv_pt : places_to_invert[i + 1]]

        if np.size(places_to_invert) % 2 != 0:
            index_sorted[places_to_invert[-1] :] = indx[places_to_invert[-1] :][::-1]
        else:
            index_sorted[places_to_invert[-1] :] = indx[places_to_invert[-1] :]
        return order1[index_sorted]
    else:
        return order1


class SchemaConverter:
    """
    Record how to convert an observation array to the standard
    opsim schema
    """

    def __init__(self):
        # Conversion dictionary, keys are opsim schema, values
        # are observation dtype names
        self.convert_dict = {
            "observationId": "ID",
            "night": "night",
            "observationStartMJD": "mjd",
            "observationStartLST": "lmst",
            "numExposures": "nexp",
            "visitTime": "visittime",
            "visitExposureTime": "exptime",
            "proposalId": "survey_id",
            "fieldId": "field_id",
            "fieldRA": "RA",
            "fieldDec": "dec",
            "altitude": "alt",
            "azimuth": "az",
            "band": "band",
            "filter": "filter",
            "airmass": "airmass",
            "skyBrightness": "skybrightness",
            "cloud": "clouds",
            "seeingFwhm500": "FWHM_500",
            "seeingFwhmGeom": "FWHM_geometric",
            "seeingFwhmEff": "FWHMeff",
            "fiveSigmaDepth": "fivesigmadepth",
            "slewTime": "slewtime",
            "slewDistance": "slewdist",
            "paraAngle": "pa",
            "pseudoParaAngle": "pseudo_pa",
            "rotTelPos": "rotTelPos",
            "rotTelPos_backup": "rotTelPos_backup",
            "rotSkyPos": "rotSkyPos",
            "rotSkyPos_desired": "rotSkyPos_desired",
            "moonRA": "moonRA",
            "moonDec": "moonDec",
            "moonAlt": "moonAlt",
            "moonAz": "moonAz",
            "moonDistance": "moonDist",
            "moonPhase": "moonPhase",
            "sunAlt": "sunAlt",
            "sunAz": "sunAz",
            "solarElong": "solarElong",
            "note": "note",
            "scheduler_note": "scheduler_note",
            "target_name": "target_name",
            "science_program": "science_program",
            "observation_reason": "observation_reason",
        }
        # For backwards compatibility
        self.backwards = {"target": "target_name"}
        # Column(s) not bothering to remap:
        # 'observationStartTime': None,
        self.inv_map = {v: k for k, v in self.convert_dict.items()}
        # angles to convert
        self.angles_rad2deg = [
            "fieldRA",
            "fieldDec",
            "altitude",
            "azimuth",
            "slewDistance",
            "paraAngle",
            "pseudoParaAngle",
            "rotTelPos",
            "rotSkyPos",
            "rotSkyPos_desired",
            "rotTelPos_backup",
            "moonRA",
            "moonDec",
            "moonAlt",
            "moonAz",
            "moonDistance",
            "sunAlt",
            "sunAz",
            "sunRA",
            "sunDec",
            "solarElong",
            "cummTelAz",
        ]
        # Put LMST into degrees too
        self.angles_hours2deg = ["observationStartLST"]

    def obs2opsim(self, obs_array, filename=None, info=None, delete_past=False, if_exists="append"):
        """Convert an array of observations into a pandas dataframe
        with Opsim schema.
        Parameters
        ----------
        obs_array : `np.array`
            Numpy array with OpSim observations.
        filename : `str`, optional
            Name of the database file to write to.
        info : `np.array`, optional
            Numpy array with database info.
        delete_past : `bool`
            Delete past observations (default=False)?
        if_exists : `str`
            Flag to pass to `to_sql` when writting to the
            database to control strategy when the database
            already exists.
        Returns
        -------
        `pd.DataFrame` or `None`
            Either the converted dataframe or `None`, if
            filename is provided.
        """
        if delete_past:
            try:
                os.remove(filename)
            except OSError:
                pass

        df = pd.DataFrame(obs_array)
        # TODO : Remove this hack which is for use with ts_scheduler
        # version <=v2.3 .. remove ts_scheduler actually drops "note".
        df.drop("note", axis=1, inplace=True)
        df = df.rename(index=str, columns=self.inv_map)
        for colname in self.angles_rad2deg:
            df[colname] = np.degrees(df[colname])
        for colname in self.angles_hours2deg:
            df[colname] = df[colname] * 360.0 / 24.0

        if filename is not None:
            con = sqlite3.connect(filename)
            df.to_sql("observations", con, index=False, if_exists=if_exists)
            if info is not None:
                df = pd.DataFrame(info)
                df.to_sql("info", con, if_exists=if_exists)
        else:
            return df

    def opsimdf2obs(self, df) -> np.recarray:
        """convert an opsim schema dataframe into an observation array.

        Parameters
        ----------
        df : `pd.DataFrame`
            Data frame containing opsim output observations.

        Returns
        -------
        obs : `np.recarray`
            Numpy array with OpSim observations.
        """
        # Do not modify the passed DataFrame, and avoid pandas getting
        # upset if a view is passed in.
        df = df.copy()

        # Make it backwards compatible if there are
        # columns that have changed names
        for key in self.backwards:
            if key in df.columns:
                df = df.rename(index=str, columns={key: self.backwards[key]})

        for key in self.angles_rad2deg:
            try:
                df[key] = np.radians(df[key])
            except (KeyError, TypeError):
                df[key] = np.nan
        for key in self.angles_hours2deg:
            try:
                df[key] = df[key] * 24.0 / 360.0
            except (KeyError, TypeError):
                df[key] = np.nan

        df = df.rename(index=str, columns=self.convert_dict)

        blank = ObservationArray()
        final_result = np.empty(df.shape[0], dtype=blank.dtype)
        # XXX-ugh, there has to be a better way.
        for key in final_result.dtype.names:
            if key in df.columns:
                final_result[key] = df[key].values
            else:
                warnings.warn(f"Column {key} not found.")

        return final_result

    def opsim2obs(self, filename):
        """convert an opsim schema dataframe into an observation array.

        Parameters
        ----------
        filename : `str`
            Sqlite file containing opsim output observations.
        """

        con = sqlite3.connect(filename)
        df = pd.read_sql("select * from observations;", con)
        return self.opsimdf2obs(df)


def hp_kd_tree(nside=DEFAULT_NSIDE, leafsize=100, scale=1e5):
    """
    Generate a KD-tree of healpixel locations

    Parameters
    ----------
    nside : int
        A valid healpix nside
    leafsize : int (100)
        Leafsize of the kdtree

    Returns
    -------
    tree : scipy kdtree
    """

    hpid = np.arange(hp.nside2npix(nside))
    ra, dec = _hpid2_ra_dec(nside, hpid)
    return _build_tree(ra, dec, leafsize, scale=scale)


class HpInLsstFov:
    """Return the healpixels in an underlying healpix grid that
    overlap an observation/pointing.

    This uses a very simple circular LSST camera model with no chip/raft gaps.

    Parameters
    ----------
    nside : `int`, optional
        Nside to match for the healpix array.
        Default None uses `set_default_nside`.
    fov_radius : `float`, optional
        Radius of the field of view in degrees. Default 1.75
        covers the inscribed circle.
    scale : `float`, optional
        How many sig figs to round when considering matches to healpixels.
        Useful for ensuring identical results cross-ploatform where
        float precision can vary.


    Examples
    --------
    Set up the class, then call to convert pointings to indices in the
    healpix array. Note that RA and dec should be in RADIANS.

    ```
    >>> ra = np.radians(30)
    >>> dec = np.radians(-20)
    >>> pointing2indx = HpInLsstFov()
    >>> indices = pointing2indx(ra, dec)
    >>> indices
    [8138, 8267]
    ```

    """

    def __init__(self, nside=DEFAULT_NSIDE, fov_radius=1.75, scale=1e5):
        self.tree = hp_kd_tree(nside=nside, scale=scale)
        self.radius = np.round(xyz_angular_radius(fov_radius) * scale).astype(int)
        self.scale = scale

    def __call__(self, ra, dec, **kwargs):
        """
        Parameters
        ----------
        ra : float, array
            RA in radians
        dec : float, array
            Dec in radians

        Returns
        -------
        indx : numpy array
            The healpixels that are within the FoV
        """
        x, y, z = _xyz_from_ra_dec(ra, dec)
        x = np.round(x * self.scale).astype(int)
        y = np.round(y * self.scale).astype(int)
        z = np.round(z * self.scale).astype(int)

        if np.size(x) == 1:
            indices = self.tree.query_ball_point((np.max(x), np.max(y), np.max(z)), self.radius)
        else:
            indices = self.tree.query_ball_point(np.vstack([x, y, z]).T, self.radius)
        return indices


class HpInComcamFov:
    """
    Return the healpixels within a ComCam pointing. Simple camera model
    with no chip gaps.
    """

    def __init__(self, nside=DEFAULT_NSIDE, side_length=0.7, scale=1e5):
        """
        Parameters
        ----------
        side_length : float (0.7)
            The length of one side of the square field of view (degrees).
        """
        self.nside = nside
        self.scale = scale
        self.tree = hp_kd_tree(nside=nside, scale=scale)
        self.side_length = np.round(np.radians(side_length * scale)).astype(int)
        self.inner_radius = np.round(xyz_angular_radius(side_length / 2.0) * scale).astype(int)
        self.outter_radius = np.round(xyz_angular_radius(side_length / 2.0 * np.sqrt(2.0)) * scale).astype(
            int
        )
        # The positions of the raft corners, unrotated
        self.corners_x = np.array(
            [
                -self.side_length / 2.0,
                -self.side_length / 2.0,
                self.side_length / 2.0,
                self.side_length / 2.0,
            ]
        )
        self.corners_y = np.array(
            [
                self.side_length / 2.0,
                -self.side_length / 2.0,
                -self.side_length / 2.0,
                self.side_length / 2.0,
            ]
        )

    def __call__(self, ra, dec, rotSkyPos=0.0):
        """
        Parameters
        ----------
        ra : float
            RA in radians
        dec : float
            Dec in radians
        rotSkyPos : float
            The rotation angle of the camera in radians
        Returns
        -------
        indx : numpy array
            The healpixels that are within the FoV
        """
        x, y, z = _xyz_from_ra_dec(ra, dec)
        x = np.round(x * self.scale).astype(int)
        y = np.round(y * self.scale).astype(int)
        z = np.round(z * self.scale).astype(int)
        # Healpixels within the inner circle
        indices = self.tree.query_ball_point((x, y, z), self.inner_radius)
        # Healpixels withing the outer circle
        indices_all = np.array(self.tree.query_ball_point((x, y, z), self.outter_radius))
        # Only need to check pixel if it is outside inner circle
        indices_to_check = indices_all[np.isin(indices_all, indices, invert=True)]

        if np.size(indices_to_check) == 0:
            ValueError("No HEALpix in pointing. Maybe need to increase nside.")

        cos_rot = np.cos(rotSkyPos)
        sin_rot = np.sin(rotSkyPos)
        x_rotated = self.corners_x * cos_rot - self.corners_y * sin_rot
        y_rotated = self.corners_x * sin_rot + self.corners_y * cos_rot

        # Draw the square that we want to check if points are in.
        bb_path = mplPath.Path(
            np.array(
                [
                    [x_rotated[0], y_rotated[0]],
                    [x_rotated[1], y_rotated[1]],
                    [x_rotated[2], y_rotated[2]],
                    [x_rotated[3], y_rotated[3]],
                    [x_rotated[0], y_rotated[0]],
                ]
            ).astype(int)
        )

        ra_to_check, dec_to_check = _hpid2_ra_dec(self.nside, indices_to_check)

        # Project the indices to check to the tangent plane, see
        # if they fall inside the polygon
        x, y = gnomonic_project_toxy(ra_to_check, dec_to_check, ra, dec)
        x = (x * self.scale).astype(int)
        y = (y * self.scale).astype(int)
        for i, xcheck in enumerate(x):
            # I wonder if I can do this all at once rather than a loop?
            if bb_path.contains_point((x[i], y[i])):
                indices.append(indices_to_check[i])

        return np.array(indices)


def run_info_table(observatory, extra_info=None):
    """
    Make a little table for recording the information about a run
    """

    observatory_info = observatory.get_info()
    if extra_info is not None:
        for key in extra_info:
            observatory_info.append([key, extra_info[key]])
    observatory_info = np.array(observatory_info)

    n_feature_entries = 3

    names = ["Parameter", "Value"]
    dtypes = ["|U200", "|U200"]
    result = np.zeros(observatory_info[:, 0].size + n_feature_entries, dtype=list(zip(names, dtypes)))

    # Fill in info about the run
    result[0]["Parameter"] = "Date, ymd"
    now = datetime.datetime.now()
    result[0]["Value"] = "%i, %i, %i" % (now.year, now.month, now.day)

    result[1]["Parameter"] = "hostname"
    result[1]["Value"] = socket.gethostname()

    result[2]["Parameter"] = "rubin_scheduler.__version__"
    result[2]["Value"] = rsVersion

    result[3:]["Parameter"] = observatory_info[:, 0]
    result[3:]["Value"] = observatory_info[:, 1]

    return result


def inrange(inval, minimum=-1.0, maximum=1.0):
    """
    Make sure values are within min/max
    """
    inval = np.array(inval)
    below = np.where(inval < minimum)
    inval[below] = minimum
    above = np.where(inval > maximum)
    inval[above] = maximum
    return inval


def warm_start(scheduler, observations, mjd_key="mjd"):
    """Replay a list of observations into the scheduler

    Parameters
    ----------
    scheduler : scheduler object
    observations : np.array
        An array of observation (e.g., from sqlite2observations)
    """

    # Check that observations are in order
    observations.sort(order=mjd_key)
    for observation in observations:
        scheduler.add_observation(observation)

    return scheduler


def season_calc(night, offset=0, modulo=None, max_season=None, season_length=365.25, floor=True):
    """
    Compute what season a night is in with possible offset and modulo
    using convention that night -365 to 0 is season -1.

    Parameters
    ----------
    night : int or array
        The night we want to convert to a season
    offset : float or array (0)
        Offset to be applied to night (days)
    modulo : int (None)
        If the season should be modulated (i.e., so we can get all
        even years) (seasons, years w/default season_length)
    max_season : int (None)
        For any season above this value (before modulo), set to -1
    season_length : float (365.25)
        How long to consider one season (nights)
    floor : bool (True)
        If true, take the floor of the season. Otherwise, returns
        season as a float
    """
    if np.size(night) == 1:
        night = np.ravel(np.array([night]))
    result = night + offset
    result = result / season_length
    if floor:
        result = np.floor(result)
    if max_season is not None:
        over_indx = np.where(IntRounded(result) >= IntRounded(max_season))

    if modulo is not None:
        neg = np.where(IntRounded(result) < IntRounded(0))
        result = result % modulo
        result[neg] = -1
    if max_season is not None:
        result[over_indx] = -1
    if floor:
        result = result.astype(int)
    return result


def create_season_offset(nside, sun_ra_rad):
    """
    Make an offset map so seasons roll properly
    """
    hpindx = np.arange(hp.nside2npix(nside))
    ra, dec = _hpid2_ra_dec(nside, hpindx)
    offset = ra - sun_ra_rad + 2.0 * np.pi
    offset = offset % (np.pi * 2)
    offset = offset * 365.25 / (np.pi * 2)
    offset = -offset - 365.25
    return offset


class TargetoO:
    """Class to hold information about a target of opportunity object

    Parameters
    ----------
    tooid : `int`
        Unique ID for the ToO.
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
    """

    def __init__(
        self,
        tooid,
        footprint,
        mjd_start,
        duration,
        ra_rad_center=None,
        dec_rad_center=None,
        too_type=None,
        posterior_distance=None,
    ):
        self.footprint = footprint
        self.duration = duration
        self.id = tooid
        self.mjd_start = mjd_start
        self.ra_rad_center = ra_rad_center
        self.dec_rad_center = dec_rad_center
        self.too_type = too_type
        self.posterior_distance = posterior_distance


class SimTargetooServer:
    """Wrapper to deliver a targetoO object at the right time"""

    def __init__(self, targeto_o_list):
        self.targeto_o_list = targeto_o_list
        self.mjd_starts = np.array([too.mjd_start for too in self.targeto_o_list])
        durations = np.array([too.duration for too in self.targeto_o_list])
        self.mjd_ends = self.mjd_starts + durations

    def __call__(self, mjd):
        in_range = np.where((mjd > self.mjd_starts) & (mjd < self.mjd_ends))[0]
        result = None
        if in_range.size > 0:
            result = [self.targeto_o_list[i] for i in in_range]
        return result
