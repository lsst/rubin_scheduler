__all__ = ("SkyModelPre", "interp_angle")

import abc
import glob
import os
import urllib
import warnings
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time

import rubin_scheduler.data.rs_download_sky
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import SURVEY_START_MJD, Site, _angular_separation, _hpid2_ra_dec

try:
    from lsst.resources import ResourcePath
except ImportError:
    pass


def short_angle_dist(a0, a1):
    """
    from https://gist.github.com/shaunlebron/8832585
    """
    max_angle = 2.0 * np.pi
    da = (a1 - a0) % max_angle
    return 2.0 * da % max_angle - da


def interp_angle(x_out, xp, anglep, degrees=False):
    """
    Interpolate angle values (handle wrap around properly).
    Does nearest neighbor interpolation if values out of range.

    Parameters
    ----------
    x_out : `float` or array
        The points to interpolate to.
    xp : array
        Points to interpolate between (must be sorted)
    anglep : array
        The angles ascociated with xp
    degrees : `bool` (False)
        Set if anglep is degrees (True) or radidian (False)
    """

    # Where are the interpolation points
    x = np.atleast_1d(x_out)
    left = np.searchsorted(xp, x) - 1
    right = left + 1

    # If we are out of bounds, just use the edges
    right[np.where(right >= xp.size)] -= 1
    left[np.where(left < 0)] += 1
    baseline = xp[right] - xp[left]

    wterm = (x - xp[left]) / baseline
    wterm[np.where(baseline == 0)] = 0
    if degrees:
        result = (
            np.radians(anglep[left])
            + short_angle_dist(np.radians(anglep[left]), np.radians(anglep[right])) * wterm
        )
        result = result % (2.0 * np.pi)
        result = np.degrees(result)
    else:
        result = anglep[left] + short_angle_dist(anglep[left], anglep[right]) * wterm
        result = result % (2.0 * np.pi)
    return result


def simple_daytime(sky_alt, sky_az, sun_alt, sun_az, band_name="r", bright_val=2.0, sky_alt_min=20.0):
    """A simple function to return a sky brightness map when the sun is up.

    The map returned is simply "bright_val" regardless of band or altitude.

    Parameters
    ----------
    sky_alt : `float`
        Altitude of poistion(s) on the sky. Degrees.
    sky_az : `float`
        Azimuth of poistion(s) on the sky. Degrees.
    sun_alt : `float`
        Altitude of the sun. Degrees.
    sun_az : `float`
        Azimuth of the sun. Degrees.
    band_name : `str`
        Name of the band, default "r". Currently unused, but
        could be useful in the future if a more complicated function
        gets subbed in.
    bright_val : `float`
        The value to plug into the sky. Default 2 mag/sq arcsec.
    sky_alt_min : `float`
        Set all sky below the alt limit to NaN. Default 20 degrees.
    """

    result = np.full_like(sky_alt, np.nan)
    result[np.where(sky_alt > sky_alt_min)] = bright_val
    return result


class SkyModelPreBase(abc.ABC):
    """Load pre-computed sky brighntess maps for the LSST site
    and use them to interpolate to arbitrary dates.

    Parameters
    ----------
    data_path : `str`, opt
        path to the numpy save files. Looks in standard
        SIMS_SKYBRIGHTNESS_DATA or RUBIN_SIM_DATA_DIR if set to
        default (None).
    init_load_length : `int` (10)
        The length of time (days) to load from disk initially.
        Set to something small for fast reads.
    load_length : `int` (365)
        The number of days to load after the initial load.
    mjd0 : `float` (None)
        The starting MJD to load on initilization (days). Uses
        util to lookup default if None.
    location : `astropy.EarthLocation`
        The location of the telescope. Default of None will load
        Rubin position.
    sun_alt_limit : `float`
            The altitude limit to use a "bright" sky function. Default
            of -8 degrees
    """

    def __init__(
        self,
        data_path=None,
        init_load_length=10,
        load_length=365,
        verbose=False,
        mjd0=None,
        location=None,
        sun_alt_limit=-8.0,
    ):
        self.info = None
        self.sb = None
        self.header = None
        self.band_names = None
        self.verbose = verbose
        self.sun_alt_limit = np.radians(sun_alt_limit)

        if location is None:
            site = Site("LSST")
            self.location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
        else:
            self.location = location

        # Look in default location for .npz files to load
        if data_path is not None:
            self.data_path = data_path
        elif "SIMS_SKYBRIGHTNESS_DATA" in os.environ:
            self.data_path = os.environ["SIMS_SKYBRIGHTNESS_DATA"]
        else:
            self.data_path = os.path.join(get_data_dir(), "skybrightness_pre")

        self._init_files()

        if len(self.files) == 0:
            errmssg = "Failed to find pre-computed .h5 files. "
            errmssg += "Copy data from NCSA with sims_skybrightness_pre/data/data_down.sh \n"
            errmssg += "or build by running sims_skybrightness_pre/data/generate_hdf5.py"
            warnings.warn(errmssg)

        self._init_filesizes()
        self._init_file_mjd_ranges()

        # Set that nothing is loaded at this point
        self.loaded_range = np.array([-1])
        self.timestep_max = -1

        if mjd0 is None:
            mjd0 = SURVEY_START_MJD

        # Do a quick initial load if set
        if init_load_length is not None:
            self.load_length = init_load_length
            self._load_data(mjd0)
        # swap back to the longer load length
        self.load_length = load_length
        self.nside = 32
        hpid = np.arange(hp.nside2npix(self.nside))
        self.ra, self.dec = _hpid2_ra_dec(self.nside, hpid)
        self.skycoord = SkyCoord(ra=self.ra * u.rad, dec=self.dec * u.rad)

    def _load_data(self, mjd, filename=None, npyfile=None):
        """Load up the h5 file to interpolate things.

        Parameters
        ----------
        mjd : `float`
            The Modified Julian Date that we want to load
        filename : `str` (None)
            The filename to restore. If None, it checks the filenames
            on disk to find one that should have the requested MJD
        npyfile : `str` (None)
            If sky brightness data not in npz file, checks the .npy
            file with same root name.
        """

        if filename is None:
            # Figure out which file to load.
            file_indx = np.where((mjd >= self.mjd_left) & (mjd <= self.mjd_right))[0]
            if np.size(file_indx) == 0:
                raise ValueError(
                    "MJD = %f is out of range for the files found (%f-%f)"
                    % (mjd, self.mjd_left.min(), self.mjd_right.max())
                )
            # Just take the later one, assuming we're probably
            # simulating forward in time
            file_indx = np.max(file_indx)

            filename = self.files[file_indx]
        else:
            self.loaded_range = None

        # Use three separate try/excepet blocks so that if any of
        # them throw exceptions, we still get the others.
        try:
            del self.sb
        except AttributeError:
            pass

        try:
            del self.band_names
        except AttributeError:
            pass

        try:
            del self.timestep_max
        except AttributeError:
            pass

        if self.verbose:
            print("Loading file %s" % filename)
        h5 = self._create_h5(filename, "r")
        mjds = h5["mjds"][:]
        indxs = np.where((mjds >= mjd) & (mjds <= (mjd + self.load_length)))
        indxs = [np.min(indxs), np.max(indxs)]
        if indxs[0] > 0:
            indxs[0] -= 1
        self.loaded_range = np.array([mjds[indxs[0]], mjds[indxs[1]]])
        self.mjds = mjds[indxs[0] : indxs[1]]
        _timestep_max = np.empty(1, dtype=float)
        h5["timestep_max"].read_direct(_timestep_max)
        self.timestep_max = np.max(_timestep_max)

        self.sb = h5["sky_mags"][indxs[0] : indxs[1]]
        self.band_names = self.sb.dtype.names
        h5.close()

        if self.verbose:
            print("%s loaded" % os.path.split(filename)[1])

        self.nside = hp.npix2nside(self.sb[self.band_names[0]][0, :].size)

    def return_mags(
        self,
        mjd,
        indx=None,
        badval=hp.UNSEEN,
        bands=["u", "g", "r", "i", "z", "y"],
        extrapolate=False,
        filters=None,
    ):
        """Return a full sky map or individual pixels for the input mjd.

        Parameters
        ----------
        mjd : `float`
            Modified Julian Date to interpolate to
        indx : `List` of `int` (None)
            indices to interpolate the sky values at. Returns full sky
            if None. If the class was instatiated with opsimFields,
            indx is the field ID, otherwise it is the healpix ID.
        airmass_mask : `bool` (True)
            Set high (>2.5) airmass pixels to badval.
        planet_mask : `bool` (True)
            Set sky maps to badval near (2 degrees) bright planets.
        moon_mask : `bool` (True)
            Set sky maps near (10 degrees) the moon to badval.
        zenith_mask : `bool` (True)
            Set sky maps at high altitude (>86.5) to badval.
        badval : `float` (-1.6375e30)
            Mask value. Defaults to the healpy mask value.
        bands : `list`, opt
            List of strings for the bands that should be returned.
            Default returns ugrizy.
        extrapolate : `bool` (False)
            In indx is set, extrapolate any masked pixels to be the
            same as the nearest non-masked value from the full sky map.
        filters : `list`, opt
            Deprecated version of bands.

        Returns
        -------
        sbs : `dict`
            A dictionary with band names as keys and np.arrays as
            values which hold the sky brightness maps in mag/sq arcsec.
        """
        if filters is not None:
            warnings.warn("filters deprecated in favor of bands", FutureWarning)
            bands = filters
        if mjd < self.loaded_range.min() or (mjd > self.loaded_range.max()):
            self._load_data(mjd)

        left = np.searchsorted(self.mjds, mjd) - 1
        right = left + 1

        # Do full sky by default
        if indx is None:
            indx = np.arange(self.sb["r"].shape[1])
            full_sky = True
        else:
            full_sky = False

        # If we are out of bounds
        if right >= self.mjds.size:
            right -= 1
            baseline = 1.0
        elif left < 0:
            left += 1
            baseline = 1.0
        else:
            baseline = self.mjds[right] - self.mjds[left]

        # Check if we are between sunrise/set
        if baseline > self.timestep_max + 1e-6:
            # Check if sun is really high:
            obstime = Time(mjd, format="mjd")
            sun = get_sun(obstime)
            aa = AltAz(location=self.location, obstime=obstime)
            sun_alt_az = sun.transform_to(aa)

            if sun_alt_az.alt.rad > self.sun_alt_limit:
                warnings.warn("Sun high, using bright sky approx")
                hp_aa = self.skycoord.transform_to(aa)

                sbs = {}
                for band_name in bands:
                    sbs[band_name] = simple_daytime(
                        hp_aa.alt.deg,
                        hp_aa.az.deg,
                        sun_alt_az.alt.deg,
                        sun_alt_az.az.deg,
                        band_name=band_name,
                    )[indx]

            else:
                warnings.warn("Requested MJD between sunrise and sunset, returning closest maps")
                diff = np.abs(self.mjds[left.max() : right.max() + 1] - mjd)
                closest_indx = np.array([left, right])[np.where(diff == np.min(diff))].min()
                sbs = {}
                for band_name in bands:
                    sbs[band_name] = self.sb[band_name][closest_indx, indx]
                    sbs[band_name][np.isinf(sbs[band_name])] = badval
                    sbs[band_name][np.where(sbs[band_name] == hp.UNSEEN)] = badval
        else:
            wterm = (mjd - self.mjds[left]) / baseline
            w1 = 1.0 - wterm
            w2 = wterm
            sbs = {}
            for band_name in bands:
                sbs[band_name] = self.sb[band_name][left, indx] * w1 + self.sb[band_name][right, indx] * w2
        # If requested a certain pixel(s), and want to extrapolate.
        if (not full_sky) & extrapolate:
            masked_pix = False
            for band_name in bands:
                if (badval in sbs[band_name]) | (True in np.isnan(sbs[band_name])):
                    masked_pix = True
            if masked_pix:
                # We have pixels that are masked that we want
                # reasonable values for
                full_sky_sb = self.return_mags(
                    mjd,
                    bands=bands,
                )
                good = np.where((full_sky_sb[bands[0]] != badval) & ~np.isnan(full_sky_sb[bands[0]]))[0]
                ra_full = self.ra[good]
                dec_full = self.dec[good]
                for bandname in bands:
                    full_sky_sb[bandname] = full_sky_sb[bandname][good]
                # Going to assume the masked pixels are the same in all bands
                masked_indx = np.where((sbs[bands[0]].ravel() == badval) | np.isnan(sbs[bands[0]].ravel()))[0]
                for i, mi in enumerate(masked_indx):
                    # Note, this is going to be really slow for many
                    # pixels, should use a kdtree
                    dist = _angular_separation(
                        self.ra[indx[i]],
                        self.dec[indx[i]],
                        ra_full,
                        dec_full,
                    )
                    closest = np.where(dist == dist.min())[0]
                    for bandname in bands:
                        sbs[bandname].ravel()[mi] = np.min(full_sky_sb[bandname][closest])

        return sbs

    @abc.abstractmethod
    def _init_files(self):
        pass

    @abc.abstractmethod
    def _init_filesizes(self):
        pass

    @abc.abstractmethod
    def _init_file_mjd_ranges(self):
        pass

    @abc.abstractmethod
    def _create_h5(self, filename, *args, **kwargs):
        pass


class SkyModelPreWithLocalFilesOnly(SkyModelPreBase):
    def _init_files(self):
        self.files = glob.glob(os.path.join(self.data_path, "*.h5"))
        self.files.sort()

    def _init_filesizes(self):
        self.filesizes = np.array([os.path.getsize(filename) for filename in self.files])

    def _init_file_mjd_ranges(self):
        mjd_left = []
        mjd_right = []
        for filename in self.files:
            temp = os.path.split(filename)[-1].replace(".h5", "").split("_")
            mjd_left.append(float(temp[0]))
            mjd_right.append(float(temp[1]))

        self.mjd_left = np.array(mjd_left)
        self.mjd_right = np.array(mjd_right)

    def _create_h5(self, filename, *args, **kwargs):
        # This method exists so it can be overriden in a subclass
        h5 = h5py.File(filename, *args, **kwargs)
        return h5


class SkyModelPreWithResources(SkyModelPreBase):
    def _init_files(self):
        data_resource = ResourcePath(self.data_path, forceDirectory=True)

        try:
            self.files = list(ResourcePath.findFileResources([data_resource], file_filter=r".*\.h5"))
        except NotImplementedError:
            # lsst.requests does not implement walk for plain html,
            # so do it manually here.
            # Unlike the method above, however, this simple approach
            # does not descend into subdirectories.
            self.files = []
            if urllib.parse.urlparse(data_resource.geturl()).scheme in ("http", "https"):
                request_content = requests.get(data_resource.geturl()).text
                html_parser = rubin_scheduler.data.rs_download_sky.MyHTMLParser()
                html_parser.feed(request_content)
                html_parser.close()
                for file_name in html_parser.filenames:
                    if file_name.endswith(".h5"):
                        self.files.append(data_resource.join(file_name))
            else:
                raise

        self.files.sort()

    def _init_filesizes(self):
        self.filesizes = np.array([file_path.size() for file_path in self.files])

    def _init_file_mjd_ranges(self):
        mjd_left = []
        mjd_right = []
        for file_resource_path in self.files:
            temp = Path(file_resource_path.split()[1]).stem.split("_")
            mjd_left.append(float(temp[0]))
            mjd_right.append(float(temp[1]))

        self.mjd_left = np.array(mjd_left)
        self.mjd_right = np.array(mjd_right)

    def _create_h5(self, filename, *args, **kwargs):
        with filename.as_local() as local_file_resource_path:
            h5 = h5py.File(local_file_resource_path.ospath, *args, **kwargs)
        return h5


if "ResourcePath" in dir():

    class SkyModelPre(SkyModelPreWithResources):
        pass

else:

    class SkyModelPre(SkyModelPreWithLocalFilesOnly):
        pass
