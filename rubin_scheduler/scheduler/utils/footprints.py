"""Footprints: Take sky area maps and turn them into dynamic `footprint`
objects which understand seasons and time, in order to weight area on sky
appropriately for a given time.
"""

__all__ = (
    "calc_norm_factor",
    "calc_norm_factor_array",
    "ecliptic_area",
    "StepLine",
    "Footprints",
    "Footprint",
    "StepSlopes",
    "ConstantFootprint",
    "BasePixelEvolution",
    "slice_wfd_area_quad",
    "slice_wfd_indx",
    "slice_quad_galactic_cut",
    "make_rolling_footprints",
    "PerFilterStep",
)

import warnings

import healpy as hp
import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.coordinates import SkyCoord

from rubin_scheduler.utils import DEFAULT_NSIDE, _hpid2_ra_dec

from .sky_area import CurrentAreaMap


def ecliptic_area(
    nside: int = DEFAULT_NSIDE,
    dist_to_eclip: float = 40.0,
    dec_max: float = 30.0,
    mask: npt.NDArray = 1,
) -> npt.NDArray:
    """Generate a HEALpix map for the area around the ecliptic

    Parameters
    ----------
    nside : `int`
        The HEALpix nside to use
    dist_to_eclip : `float`
        The distance to the ecliptic to constrain to (degrees).
        Default 40.
    dec_max : `float`
        The max declination to alow (degrees).
        Default 30.
    mask : `np.array`
        Any additional mask to apply, should be a
        HEALpix mask with matching nside. Default None.
    """

    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
    result = np.zeros(ra.size)
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    eclip_lat = coord.barycentrictrueecliptic.lat.radian
    good = np.where((np.abs(eclip_lat) < np.radians(dist_to_eclip)) & (dec < np.radians(dec_max)))
    result[good] += 1

    result *= mask

    return result


def make_rolling_footprints(
    fp_hp=None,
    mjd_start=60218.0,
    sun_ra_start=3.27717639,
    nslice=2,
    scale=0.8,
    nside=DEFAULT_NSIDE,
    wfd_indx=None,
    order_roll=0,
    n_cycles=None,
    n_constant_start=2,
    n_constant_end=6,
    verbose=False,
    uniform=True,
):
    """
    Generate rolling footprints

    Parameters
    ----------
    fp_hp : dict-like
        A dict with band name keys and HEALpix map values.
        Default None will load CurrentAreaMap. Assumes
        WFD is where r-band is 1.
    mjd_start : `float`
        The starting date of the survey.
    sun_ra_start : `float`
        The RA of the sun at the start of the survey
    nslice : `int`
        How much to slice the sky up. Can be 2, 3, 4, or 6.
    scale : `float`
        The strength of the rolling, value of 1 is full power rolling.
        Zero is no rolling.
    wfd_indx : array of ints
        The indices of the HEALpix map that are to be included in the rolling.
    order_roll : `int`
        Change the order of when bands roll. Default 0.
    n_cycles : `int`
        Number of complete rolling cycles to attempt. If None, defaults to 3
        full cycles for nslice=2, 2 cycles for nslice=3 or 4, and 1 cycle for
        nslice=6.
    n_constant_start : `int`
        The number of constant non-rolling seasons to start with. Anything
        less than 2 will start rolling too early near Y1. Defaults to 2.
    n_constant_end : `int`
        The number of constant seasons to end the survey with. Defaults to 6.

    Returns
    -------
    Footprints object
    """

    if fp_hp is None:
        sky = CurrentAreaMap(nside=nside)
        footprints, labels = sky.return_maps()
        fp_hp = {}
        for key in footprints.dtype.names:
            fp_hp[key] = footprints[key]

    nc_default = {2: 3, 3: 2, 4: 2, 6: 1}
    if n_cycles is None:
        n_cycles = nc_default[nslice]

    hp_footprints = fp_hp

    D = 1.0 - scale
    U = nslice - D * (nslice - 1)

    start = [1.0] * n_constant_start
    # After n_cycles, just go to no-rolling for 6 years.
    end = [1.0] * n_constant_end

    rolling = [U] + [D] * (nslice - 1)
    rolling = np.roll(rolling, order_roll).tolist()

    all_slopes = []
    if uniform:
        extra_constant = [1]
    else:
        extra_constant = []

    for i in range(nslice):
        _roll = np.roll(rolling, i).tolist() + extra_constant
        all_slopes.append(start + _roll * n_cycles + end)
    for i in range(nslice):
        _roll = np.roll(rolling, i).tolist() + extra_constant
        _roll = [_roll[-1]] + _roll[1:-1] + [_roll[0]]
        all_slopes.append(start + _roll * n_cycles + end)
    dvals = {
        1: "1",
        D: "D",
        U: "U",
    }

    abc = ["a", "b", "c", "d", "e", "f", "g", "h"]
    slice_names = ["slice %s" % abc[i] for i in range(nslice)]
    for i, s in enumerate(all_slopes):
        if i >= nslice:
            sname = slice_names[i - nslice] + " w/ ra - sun_ra in [90, 270]"
        else:
            sname = slice_names[i] + " w/ ra - sun_ra in [270, 90]"
        if verbose:
            print(sname + ": " + " ".join([dvals[x] for x in s]))

    fp_non_wfd = Footprint(mjd_start, sun_ra_start=sun_ra_start, nside=nside)
    rolling_footprints = []
    for i in range(len(all_slopes)):
        step_func = StepSlopes(rise=all_slopes[i])
        rolling_footprints.append(
            Footprint(
                mjd_start,
                sun_ra_start=sun_ra_start,
                step_func=step_func,
                nside=nside,
            )
        )

    wfd = hp_footprints["r"] * 0
    if wfd_indx is None:
        wfd_indx = np.where(hp_footprints["r"] == 1)[0]

    wfd[wfd_indx] = 1
    non_wfd_indx = np.where(wfd == 0)[0]

    if uniform:
        split_wfd_indices = slice_quad_galactic_cut(
            hp_footprints,
            nslice=nslice,
            wfd_indx=wfd_indx,
            ra_range=(sun_ra_start + 1.5 * np.pi, sun_ra_start + np.pi / 2),
        )

        split_wfd_indices_delayed = slice_quad_galactic_cut(
            hp_footprints,
            nslice=nslice,
            wfd_indx=wfd_indx,
            ra_range=(sun_ra_start + np.pi / 2, sun_ra_start + 1.5 * np.pi),
        )
    else:
        split_wfd_indices = slice_quad_galactic_cut(hp_footprints, nslice=nslice, wfd_indx=wfd_indx)

    for key in hp_footprints:
        temp = hp_footprints[key] + 0
        temp[wfd_indx] = 0
        fp_non_wfd.set_footprint(key, temp)

        for i in range(nslice):
            # make a copy of the current band
            temp = hp_footprints[key] + 0
            # Set the non-rolling area to zero
            temp[non_wfd_indx] = 0

            indx = split_wfd_indices[i]
            # invert the indices
            ze = temp * 0
            ze[indx] = 1
            temp = temp * ze
            rolling_footprints[i].set_footprint(key, temp)
        if uniform:
            for _i in range(nslice, nslice * 2):
                # make a copy of the current band
                temp = hp_footprints[key] + 0
                # Set the non-rolling area to zero
                temp[non_wfd_indx] = 0

                indx = split_wfd_indices_delayed[_i - nslice]
                # invert the indices
                ze = temp * 0
                ze[indx] = 1
                temp = temp * ze
                rolling_footprints[_i].set_footprint(key, temp)

    result = Footprints([fp_non_wfd] + rolling_footprints)
    return result


def _is_in_ra_range(ra, low, high):
    _low = low % (2.0 * np.pi)
    _high = high % (2.0 * np.pi)
    if _low <= _high:
        return (ra >= _low) & (ra <= _high)
    else:
        return (ra >= _low) | (ra <= _high)


def slice_quad_galactic_cut(target_map, nslice=2, wfd_indx=None, ra_range=None):
    """
    Helper function for generating rolling footprints

    Parameters
    ----------
    target_map : dict of HEALpix maps
        The final desired footprint as HEALpix maps. Keys are band names
    nslice : `int`
        The number of slices to make, can be 2 or 3.
    wfd_indx : array of ints
        The indices of target_map that should be used for rolling.
        If None, assumes the rolling area should be where target_map['r'] == 1.
    ra_range : tuple of floats, optional
        If not None, then the indices are restricted to the given RA range
        in radians.
    """
    nside = hp.npix2nside(target_map["r"].size)
    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    _, gal_lat = coord.galactic.l.deg, coord.galactic.b.deg

    indx_north = np.intersect1d(np.where(gal_lat >= 0)[0], wfd_indx)
    indx_south = np.intersect1d(np.where(gal_lat < 0)[0], wfd_indx)

    splits_north = slice_wfd_area_quad(target_map, nslice=nslice, wfd_indx=indx_north)
    splits_south = slice_wfd_area_quad(target_map, nslice=nslice, wfd_indx=indx_south)

    slice_indx = []
    for j in np.arange(nslice):
        indx_temp = []
        for i in np.arange(j + 1, nslice * 2 + 1, nslice):
            indx_temp += indx_north[splits_north[i - 1] : splits_north[i]].tolist()
            indx_temp += indx_south[splits_south[i - 1] : splits_south[i]].tolist()
        slice_indx.append(indx_temp)

    if ra_range is not None:
        ra_indx = np.where(_is_in_ra_range(ra, *ra_range))[0]
        for j in range(nslice):
            slice_indx[j] = np.intersect1d(ra_indx, slice_indx[j])

    return slice_indx


def slice_wfd_area_quad(target_map, nslice=2, wfd_indx=None):
    """
    Divide a healpix map in an intelligent way

    Parameters
    ----------
    target_map : dict of HEALpix arrays
        The input map to slice
    nslice : int
        The number of slices to divide the sky into (gets doubled).
    wfd_indx : array of int
        The indices of the healpix map to consider as part of the WFD area
        that will be split.
        If set to None, the pixels where target_map['r'] == 1 are
        considered as WFD.
    """
    nslice2 = nslice * 2

    wfd = target_map["r"] * 0
    if wfd_indx is None:
        wfd_indices = np.where(target_map["r"] == 1)[0]
    else:
        wfd_indices = wfd_indx
    wfd[wfd_indices] = 1
    wfd_accum = np.cumsum(wfd)
    split_wfd_indices = np.floor(np.max(wfd_accum) / nslice2 * (np.arange(nslice2) + 1)).astype(int)
    split_wfd_indices = split_wfd_indices.tolist()
    split_wfd_indices = [0] + split_wfd_indices

    return split_wfd_indices


def slice_wfd_indx(target_map, nslice=2, wfd_indx=None):
    """
    simple map split
    """

    wfd = target_map["r"] * 0
    if wfd_indx is None:
        wfd_indx = np.where(target_map["r"] == 1)[0]
    wfd[wfd_indx] = 1
    wfd_accum = np.cumsum(wfd)
    split_wfd_indices = np.floor(np.max(wfd_accum) / nslice * (np.arange(nslice) + 1)).astype(int)
    split_wfd_indices = split_wfd_indices.tolist()
    split_wfd_indices = [0] + split_wfd_indices

    return split_wfd_indices


class BasePixelEvolution:
    """Helper class that can be used to describe the time evolution of a
    HEALpix in a footprint.
    """

    def __init__(self, period=365.25, rise=1.0, t_start=0.0):
        self.period = period
        self.rise = rise
        self.t_start = t_start

    def __call__(self, mjd_in, phase):
        pass


class StepLine(BasePixelEvolution):
    def __call__(self, mjd_in, phase):
        t = mjd_in + phase - self.t_start
        n_periods = np.floor(t / (self.period))
        result = n_periods * self.rise
        tphased = t % self.period
        step_area = np.where(tphased > self.period / 2.0)[0]
        result[step_area] += (tphased[step_area] - self.period / 2) * self.rise / (0.5 * self.period)
        result[np.where(t < 0)] = 0
        return result


class StepSlopes(BasePixelEvolution):
    def __call__(self, mjd_in, phase):
        steps = np.array(self.rise)
        t = mjd_in + phase - self.t_start
        season = np.floor(t / (self.period))
        season = season.astype(int)
        plateus = np.cumsum(steps) - steps[0]
        result = plateus[season]
        tphased = t % self.period
        step_area = np.where(tphased > self.period / 2.0)[0]
        result[step_area] += (
            (tphased[step_area] - self.period / 2) * steps[season + 1][step_area] / (0.5 * self.period)
        )
        result[np.where(t < 0)] = 0

        return result


class Footprint:
    """An object to compute the desired survey footprint at a given time

    Parameters
    ----------
    mjd_start : `float`
        The MJD the survey starts on.
    sun_ra_start : `float`
        The RA of the sun at the start of the survey (radians).
    bands : `list` of `str`
        The band names to include in the footprint.
    filters : `list` of `str`
        Deprecated version of bands.
    period : `float`
        Used for setting the phase of step_func (days). Default 365.25.
    step_func : `BasePixelEvolution`
        Callable class that determines how the footprint evolves with time.
        Default of None will result in `StepLine` being used.
    """

    def __init__(
        self,
        mjd_start,
        sun_ra_start=0,
        nside=DEFAULT_NSIDE,
        bands=["u", "g", "r", "i", "z", "y"],
        filters=None,
        period=365.25,
        step_func=None,
    ):
        if filters is not None:
            warnings.warn("Use of `filters` will be deprecated in favor of `bands` at v4", FutureWarning)
            bands = filters
        # Dict to map band name to array index.
        if not isinstance(bands, dict):
            bands_dict = {}
            for i, bandname in enumerate(bands):
                bands_dict[bandname] = i
            bands = bands_dict
        self.period = period
        self.nside = nside
        if step_func is None:
            step_func = StepLine()
        self.step_func = step_func
        self.mjd_start = mjd_start
        self.sun_ra_start = sun_ra_start
        self.npix = hp.nside2npix(nside)
        self.bands = bands
        if filters is not None:
            self.filters = filters
        self.ra, self.dec = _hpid2_ra_dec(self.nside, np.arange(self.npix))
        # Set the phase of each healpixel.
        # If RA to sun is zero, we are at phase np.pi/2.
        # This is similar to the season calculation, except
        # that phase and season are offset by 90 degrees.
        self.phase = (-self.ra + self.sun_ra_start + np.pi / 2) % (2.0 * np.pi)
        self.phase = self.phase * (self.period / 2.0 / np.pi)
        # Empty footprints to start
        self.out_dtype = list(zip(bands, [float] * len(bands)))
        self.footprints = np.zeros((len(bands), self.npix), dtype=float)
        self.estimate = np.zeros((len(bands), self.npix), dtype=float)
        self.current_footprints = np.zeros((len(bands), self.npix), dtype=float)
        self.zero = self.step_func(0.0, self.phase)
        self.mjd_current = None

    def set_footprint(self, bandname, values):
        self.footprints[self.bands[bandname], :] = values

    def get_footprint(self, bandname):
        return self.footprints[self.bands[bandname], :]

    def _update_mjd(self, mjd, norm=True):
        if mjd != self.mjd_current:
            self.mjd_current = mjd
            t_elapsed = mjd - self.mjd_start

            norm_coverage = self.step_func(t_elapsed, self.phase)
            norm_coverage -= self.zero
            self.current_footprints = self.footprints * norm_coverage
            c_sum = np.nansum(self.current_footprints)
            if norm:
                if c_sum != 0:
                    self.current_footprints = self.current_footprints / c_sum

    def arr2struc(self, inarr):
        """Take an array and convert it to labeled struc array"""
        result = np.empty(self.npix, dtype=self.out_dtype)
        for key in self.bands:
            result[key] = inarr[self.bands[key]]
        # Argle bargel, why doesn't this view work?
        # struc = inarr.view(dtype=self.out_dtype).squeeze()
        return result

    def estimate_counts(self, mjd, nvisits=2.2e6, fov_area=9.6):
        """Estimate the counts we'll get after some time and visits"""
        pix_area = hp.nside2pixarea(self.nside, degrees=True)
        pix_per_visit = fov_area / pix_area
        self._update_mjd(mjd, norm=True)
        self.estimate = self.current_footprints * pix_per_visit * nvisits
        return self.arr2struc(self.estimate)

    def __call__(self, mjd, norm=True):
        """
        Parameters
        ----------
        mjd : `float`
            Current MJD.
        norm : `bool`
            If normalized, the footprint retains the same range of values
            over time.

        Returns
        -------
        current_footprints : `np.ndarray`, (6, N)
            A numpy structured array with the updated normalized number of
            observations that should be requested at each Healpix.
            Multiply by the number of HEALpix observations (all bands), to
            convert to the number of observations desired.
        """
        self._update_mjd(mjd, norm=norm)
        return self.arr2struc(self.current_footprints)


class ConstantFootprint(Footprint):
    def __init__(self, nside=DEFAULT_NSIDE, bands=["u", "g", "r", "i", "z", "y"], filters=None):
        if filters is not None:
            warnings.warn("Use of `filters` will be deprecated in favor of `bands` at v4", FutureWarning)
            bands = filters
        if not isinstance(bands, dict):
            bands_dict = {}
            for i, bandname in enumerate(bands):
                bands_dict[bandname] = i
            bands = bands_dict
        self.nside = nside
        self.bands = bands
        if filters is not None:
            self.filters = filters
        self.npix = hp.nside2npix(nside)
        self.footprints = np.zeros((len(bands), self.npix), dtype=float)
        self.out_dtype = list(zip(bands, [float] * len(bands)))
        self.to_return = self.arr2struc(self.footprints)

    def set_footprint(self, bandname, values):
        self.footprints[self.bands[bandname], :] = values
        self.to_return = self.arr2struc(self.footprints)

    def __call__(self, mjd, array=False):
        return self.to_return


class Footprints(Footprint):
    """An object to combine multiple Footprint objects."""

    def __init__(self, footprint_list):
        self.footprint_list = footprint_list
        self.mjd_current = None
        self.current_footprints = 0
        # Should probably run a check that all the footprints are compatible
        # (same nside, etc)
        self.npix = footprint_list[0].npix
        self.out_dtype = footprint_list[0].out_dtype
        self.bands = footprint_list[0].bands
        self.nside = footprint_list[0].nside

        self.footprints = np.zeros((len(self.bands), self.npix), dtype=float)
        for fp in self.footprint_list:
            self.footprints += fp.footprints

    def set_footprint(self, bandname, values):
        pass

    def _update_mjd(self, mjd, norm=True):
        if mjd != self.mjd_current:
            self.mjd_current = mjd
            self.current_footprints = 0.0
            for fp in self.footprint_list:
                fp._update_mjd(mjd, norm=False)
                self.current_footprints += fp.current_footprints
            c_sum = np.sum(self.current_footprints)
            if norm:
                if c_sum != 0:
                    self.current_footprints = self.current_footprints / c_sum


def calc_norm_factor(goal_dict, radius=1.75):
    """Calculate how to normalize a Target_map_basis_function.
    This is basically:
    the area of the fov / area of a healpixel  /
    the sum of all of the weighted-healpix values in the footprint.

    Parameters
    -----------
    goal_dict : dict of healpy maps
        The target goal map(s) being used
    radius : float (1.75)
        Radius of the FoV (degrees)

    Returns
    -------
    Value to use as Target_map_basis_function norm_factor kwarg
    """
    all_maps_sum = 0
    for key in goal_dict:
        good = np.where(goal_dict[key] > 0)
        all_maps_sum += goal_dict[key][good].sum()
    nside = hp.npix2nside(goal_dict[key].size)
    hp_area = hp.nside2pixarea(nside, degrees=True)
    norm_val = radius**2 * np.pi / hp_area / all_maps_sum
    return norm_val


def calc_norm_factor_array(goal_map, radius=1.75):
    """Calculate how to normalize a Target_map_basis_function.
    This is basically:
    the area of the fov / area of a healpixel  /
    the sum of all of the weighted-healpix values in the footprint.

    Parameters
    -----------
    goal_map : recarray of healpy maps
        The target goal map(s) being used
    radius : float
        Radius of the FoV (degrees)

    Returns
    -------
    Value to use as Target_map_basis_function norm_factor kwarg
    """
    all_maps_sum = 0
    for key in goal_map.dtype.names:
        good = np.where(goal_map[key] > 0)
        all_maps_sum += goal_map[key][good].sum()
    nside = hp.npix2nside(goal_map[key].size)
    hp_area = hp.nside2pixarea(nside, degrees=True)
    norm_val = radius**2 * np.pi / hp_area / all_maps_sum
    return norm_val


class PerFilterStep(BasePixelEvolution):
    """Make a custom step function per filter
    Compensate for filters not always being loaded.
    This does not use season information, so only
    useful for a survey of limited duration/limited
    RA range (e.g., a science validation survey).

    Parameters
    ----------
    survey_length : `float`
      Suervey length in days. Default 80.
    bands : `list`
        Bandnames. Defualt of none uses ugrizy
    loaded_dict : `dict`
        Dict with keys of bandnames and values of
        arrays of ints with the nights that band is
        loaded. E.g.,
        loaded_dict = {'u': np.array([1,2,3,15,16,17])}
        means the u band will only be availble on those
        nights.
    """

    def __init__(
        self,
        survey_length=80,
        bands=None,
        loaded_dict=None,
    ):

        if bands is None:
            bands = ["u", "g", "r", "i", "z", "y"]
        if loaded_dict is None:
            loaded_dict = {}

        self.bands = bands
        self.survey_length = survey_length

        self.bands2indx = {}
        for i, bandname in enumerate(bands):
            self.bands2indx[bandname] = i

        self.slopes = {}
        self.loaded_dict = {}

        for bandname in bands:
            if bandname in loaded_dict.keys():
                self.loaded_dict[bandname] = loaded_dict[bandname]
                self.slopes[bandname] = 1.0 / self.loaded_dict[bandname].size

    def __call__(self, t_elapsed, phase):
        """
        Parameters
        ----------
        t_elapsed : `float`
            Time elapsed in the survey (days).
        """

        # filters all the time evolve linearly increase between
        # 0 and 1 for length of survey.
        frac_done = t_elapsed / self.survey_length

        # broadcast out to n_filters
        result = np.tile(phase * 0 + frac_done, (len(self.bands), 1))
        for bandname in self.loaded_dict:
            days_completed = np.where(self.loaded_dict[bandname] <= t_elapsed)[0].size
            result[self.bands2indx[bandname], :] = (
                result[self.bands2indx[bandname], :] * 0 + days_completed * self.slopes[bandname]
            )

        return result
