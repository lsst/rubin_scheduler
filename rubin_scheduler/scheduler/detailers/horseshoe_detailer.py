import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.detailers import BaseDetailer
from rubin_scheduler.scheduler.utils import wrap_ra_dec
from rubin_scheduler.utils import (
    DEFAULT_NSIDE,
    _angular_separation,
    bearing,
    dest_latlon,
)


class HorseshoeAvoidanceDetailer(BaseDetailer):
    """A Rubin Scheduler Detailer to avoid stray light reflections from the L3
    horseshoe baffle.

    When a bright star falls within a specific off-axis angular annulus
    ([1.95, 2.05] degrees by default) from the field of view center,
    reflections off the L3 lens baffle create a characteristic stray light
    artifact known as the "Horseshoe".

    This Detailer intercepts candidate observations during scheduling, checks if
    any bright star from a given catalog (e.g. Yale Bright Star Catalog, YBSC v5)
    falls into the forbidden annulus using an ultra-fast 3D Cartesian KD-tree.

    The magnitude check is evaluated in the specific filter/band of observation:
    - If observing in band R, it checks whether the star has R_mag <= 3.0.
    - If observing in band I, it checks whether the star has I_mag <= 3.0.
    - And similarly for u, g, z, y.

    If a star exceeds the brightness threshold in the observation's reference
    filter, the detailer moves the pointing so the star no longer lands in the
    annulus.

    The move is adaptive: the offending star is located, and the field center
    is displaced along the great circle pointing directly away from it by the
    smallest distance that pushes the star past `outer_radius_deg`. Candidate
    displacements are re-checked with `check_pointings`, so a move is only
    accepted if it leaves no bright star in the annulus -- including stars that
    the move itself could have dragged in from the opposite side. Escapes are
    always outward, so a bright star is never pulled inward toward the science
    field of view.

    Parameters
    ----------
    catalog_path : `str` or `Path`, optional
        Path to the FITS catalog containing stars with coordinates and
        magnitudes. Defaults to `rubin_one_year_output/ybsc_v5.fits`.
    mag_limit : `float` or `dict`, optional
        Limiting magnitude threshold. If a float is passed (default 3.0), the
        same limit is applied across all bands. If a dict is passed (e.g.
        `{'r': 3.0, 'i': 3.0, 'default': 3.0}`), per-band limits are applied.
    use_observation_band : `bool`, optional
        If True (default), evaluates star brightness using the star's magnitude
        in the reference filter of the observation (u, g, r, i, z, y).
        If False, uses the standard V-band magnitude.
    inner_radius_deg : `float`, optional
        Inner radius of the forbidden annulus in degrees. Default 1.995 deg.
    outer_radius_deg : `float`, optional
        Outer radius of the forbidden annulus in degrees. Default 2.025 deg.
    escape_margin_deg : `float`, optional
        Extra distance pushed beyond `outer_radius_deg` so that a star does not
        come to rest exactly on the annulus boundary. Default 0.005 deg.
    max_offset_deg : `float` or None, optional
        Largest displacement the detailer may apply, in degrees. A star
        anywhere in the annulus is ejected by at most
        `(outer_radius_deg - inner_radius_deg) + escape_margin_deg`, so when
        None (default) the budget is twice that -- enough to clear any single
        offender, with headroom for the fallback ring search.
    n_ring_bearings : `int`, optional
        Number of evenly spaced bearings used in the fallback ring search when
        no single push-away direction clears the annulus. Default 12.
    annotate_note : `bool`, optional
        If True, appends `note_tag` to `observation_array['scheduler_note']`
        or `observation_array['note']` when an observation is shifted.
        Default True.
    log_offsets : `bool`, optional
        Log the offsets that have been applied in shifted_history
        attribute. Default False.
    note_tag : `str`, optional
        Substring tag added to the observation note. Default "hs_avoid".
    nside : `int`, optional
        HEALPix resolution parameter for the BaseDetailer inheritance.
        Controls spatial resolution for scheduler computations (default: DEFAULT_NSIDE).
        Must be a power of 2 (e.g., 32, 64, 128, 256).
        This parameter is passed from external scripts like simulate_one_year_with_detailer.py
        to support simulations at different resolution levels.
    """

    def __init__(
        self,
        catalog_path=None,
        mag_limit=3.0,
        use_observation_band=True,
        inner_radius_deg=1.95,
        outer_radius_deg=2.05,
        escape_margin_deg=0.015,
        max_offset_deg=None,
        n_ring_bearings=12,
        annotate_note=True,
        log_offsets=False,
        note_tag="hs_avoid",
        nside=DEFAULT_NSIDE,
    ):
        super().__init__(nside=nside)
        self.nside = int(nside)

        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(
                f"nside must be a positive power of 2, got {self.nside}. "
                f"Valid values: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ..."
            )

        if catalog_path is None:
            catalog_path = os.path.join(get_data_dir(), "scheduler/ybsc_v5.fits")
        self.catalog_path = Path(catalog_path)

        self.mag_limit = mag_limit

        self.use_observation_band = bool(use_observation_band)
        self.inner_radius_deg = float(inner_radius_deg)
        self.outer_radius_deg = float(outer_radius_deg)
        self.annotate_note = bool(annotate_note)
        self.note_tag = str(note_tag)

        self.log_offsets = bool(log_offsets)

        # Adaptive offset configuration
        self.escape_margin_deg = float(escape_margin_deg)
        self.escape_margin_rad = float(np.radians(self.escape_margin_deg))
        self.n_ring_bearings = int(n_ring_bearings)

        # A star anywhere in the annulus is ejected by at most
        # (outer - inner) + margin; double that gives the ring search room to
        # work when a single push-away direction is blocked.
        if max_offset_deg is None:
            max_offset_deg = 2.0 * (self.outer_radius_deg - self.inner_radius_deg + self.escape_margin_deg)
        self.max_offset_deg = float(max_offset_deg)
        self.max_offset_rad = float(np.radians(self.max_offset_deg))

        self.inner_radius_rad = float(np.radians(self.inner_radius_deg))
        self.outer_radius_rad = float(np.radians(self.outer_radius_deg))

        # Chord distances on the 3D unit sphere for the annulus:
        # chord distance d = 2 * sin(theta / 2)
        self.d_min = 2.0 * np.sin(np.radians(self.inner_radius_deg) / 2.0)
        self.d_max = 2.0 * np.sin(np.radians(self.outer_radius_deg) / 2.0)

        # Load star catalog, calculate filter band magnitudes, and build KD-tree
        self._load_catalog()

        # Cumulative execution counters and history
        self.n_evaluated = 0
        self.n_shifted = 0
        self.n_failed = 0
        self.shifted_history = []
        self.n_shifted_by_band = {b: 0 for b in ["u", "g", "r", "i", "z", "y", "other"]}

    def get_mag_limit(self, band):
        """Retrieve limiting magnitude for a given filter band."""
        if isinstance(self.mag_limit, dict):
            b_key = str(band).lower()
            return float(self.mag_limit.get(b_key, self.mag_limit.get("default", 3.0)))
        return float(self.mag_limit)

    def _load_catalog(self):
        """Load star catalog, compute multi-band magnitudes, and build KD-tree."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Star catalog FITS file not found: {self.catalog_path}")

        with fits.open(self.catalog_path) as hdul:
            data = hdul[1].data

            # Identify RA and Dec columns
            ra_candidates = ["coord_ra", "RA_DEG", "ra", "RA"]
            dec_candidates = ["coord_dec", "DEC_DEG", "dec", "DEC", "de"]

            ra_col = next((c for c in ra_candidates if c in data.names), None)
            dec_col = next((c for c in dec_candidates if c in data.names), None)

            if ra_col is None or dec_col is None:
                raise KeyError(
                    f"Could not find RA/Dec columns in {self.catalog_path}. "
                    f"Available columns: {data.names}"
                )

            mask = np.isfinite(data[ra_col]) & np.isfinite(data[dec_col])

            ra_deg = np.asarray(data[ra_col][mask], dtype=float)
            dec_deg = np.asarray(data[dec_col][mask], dtype=float)

            # Extract base Johnson magnitudes and colors
            has_vmag = "Vmag" in data.names
            has_bmv = "B-V" in data.names
            has_ub = "U-B" in data.names

            vmag = np.asarray(data["Vmag"][mask], dtype=float) if has_vmag else np.full(len(ra_deg), np.nan)
            bmv = np.asarray(data["B-V"][mask], dtype=float) if has_bmv else np.full(len(ra_deg), np.nan)
            ub = np.asarray(data["U-B"][mask], dtype=float) if has_ub else np.full(len(ra_deg), np.nan)

        self.star_mags = {"V": vmag}

        # ------------------------------------------------------------------
        # Compute band magnitudes (DES / LSST system transformations)
        # ------------------------------------------------------------------
        # g-band
        gmag = np.copy(vmag)
        m1 = (bmv > -0.2) & (bmv <= 0.4)
        m2 = (bmv > 0.4) & (bmv <= 2.2)
        gmag[m1] = vmag[m1] + 0.552 * bmv[m1] - 0.099
        gmag[m2] = vmag[m2] + 0.493 * bmv[m2] - 0.067
        self.star_mags["g"] = gmag

        # r-band
        rmag = np.copy(vmag)
        mr = (bmv > -0.2) & (bmv <= 2.2)
        rmag[mr] = vmag[mr] - 0.543 * bmv[mr] + 0.128
        self.star_mags["r"] = rmag

        # i-band
        imag = np.copy(vmag)
        mi = (bmv > -0.2) & (bmv <= 2.2)
        imag[mi] = vmag[mi] - 1.04 * bmv[mi] + 0.312
        self.star_mags["i"] = imag

        # z-band
        zmag = np.copy(vmag)
        mz = (bmv > -0.2) & (bmv <= 2.2)
        zmag[mz] = vmag[mz] - 1.302 * bmv[mz] + 0.417
        self.star_mags["z"] = zmag

        # y-band
        ymag = np.copy(vmag)
        my = (bmv > -0.2) & (bmv <= 2.2)
        ymag[my] = vmag[my] - 1.416 * bmv[my] + 0.504
        self.star_mags["y"] = ymag

        # u-band (u ≈ U_Johnson - 0.7 where U = V + B-V + U-B, fallback to Vmag)
        umag = np.copy(vmag)
        mu = np.isfinite(bmv) & np.isfinite(ub)
        umag[mu] = vmag[mu] + bmv[mu] + ub[mu] - 0.7
        self.star_mags["u"] = umag

        # Convert to 3D Cartesian unit vectors for fast spatial indexing
        self.star_ra_rad = np.radians(ra_deg)
        self.star_dec_rad = np.radians(dec_deg)

        self.xyz_stars = np.column_stack(
            [
                np.cos(self.star_dec_rad) * np.cos(self.star_ra_rad),
                np.cos(self.star_dec_rad) * np.sin(self.star_ra_rad),
                np.sin(self.star_dec_rad),
            ]
        )
        self.tree = cKDTree(self.xyz_stars)

    def check_pointings(self, ra_rad, dec_rad, bands=None):
        """Check whether pointings have stars falling in the forbidden annulus
        exceeding the limiting magnitude in the observation filter.

        Parameters
        ----------
        ra_rad : `float` or array-like
            Right Ascension in radians.
        dec_rad : `float` or array-like
            Declination in radians.
        bands : `str` or array-like, optional
            Filter band names for each pointing (e.g. 'r', 'i', 'z'). If None,
            defaults to 'V'.

        Returns
        -------
        compromised : `np.ndarray` of `bool`
            Boolean mask indicating which pointings contain a star in the
            forbidden annulus exceeding the magnitude threshold in the reference filter.
        hit_info : `list` of `list` of `dict`
            List of detected stars per pointing, with star index, magnitude in
            the observation band, V-band magnitude, and angular separation.
        """
        ra_rad = np.atleast_1d(ra_rad)
        dec_rad = np.atleast_1d(dec_rad)
        n_pts = len(ra_rad)

        if bands is None:
            band_list = ["v"] * n_pts
        elif isinstance(bands, str):
            band_list = [bands.lower()] * n_pts
        else:
            band_list = [str(b).lower() for b in bands]

        xyz_obs = np.column_stack(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ]
        )

        candidates = self.tree.query_ball_point(xyz_obs, r=self.d_max)

        compromised = np.zeros(n_pts, dtype=bool)
        hit_info = []

        for i in range(n_pts):
            cand_indices = candidates[i]
            cand_info = []
            if len(cand_indices) > 0:
                dists = np.linalg.norm(self.xyz_stars[cand_indices] - xyz_obs[i], axis=1)
                annulus_mask = dists >= self.d_min
                if np.any(annulus_mask):
                    annulus_stars = np.array(cand_indices)[annulus_mask]
                    annulus_dists = dists[annulus_mask]

                    obs_b = band_list[i]
                    if self.use_observation_band:
                        mag_ref_array = self.star_mags.get(obs_b, self.star_mags["V"])
                        mag_thresh = self.get_mag_limit(obs_b)
                    else:
                        mag_ref_array = self.star_mags["V"]
                        mag_thresh = self.get_mag_limit("V")

                    bright_in_band = mag_ref_array[annulus_stars] <= mag_thresh
                    if np.any(bright_in_band):
                        compromised[i] = True
                        matched_stars = annulus_stars[bright_in_band]
                        matched_dists = annulus_dists[bright_in_band]

                        for s_idx, d in zip(matched_stars, matched_dists):
                            # Chord distance -> great-circle separation.
                            sep_rad = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
                            # Bearing from the field center toward the star;
                            # the adaptive offset pushes along bearing + pi.
                            star_bearing = bearing(
                                ra_rad[i],
                                dec_rad[i],
                                self.star_ra_rad[s_idx],
                                self.star_dec_rad[s_idx],
                            )
                            cand_info.append(
                                {
                                    "star_idx": int(s_idx),
                                    "band": obs_b,
                                    "star_mag_band": float(mag_ref_array[s_idx]),
                                    "star_vmag": float(self.star_mags["V"][s_idx]),
                                    "mag_threshold": float(mag_thresh),
                                    "sep_deg": float(np.degrees(sep_rad)),
                                    "sep_rad": float(sep_rad),
                                    "bearing_rad": float(star_bearing),
                                }
                            )
            hit_info.append(cand_info)

        return compromised, hit_info

    def _candidate_offsets(self, stars):
        """Build candidate great-circle displacements to escape the annulus.

        Each candidate is a `(dist_rad, bearing_rad)` pair describing a move of
        the field center. Moving a distance `d` along `bearing(center -> star)
        + pi` increases the center-star separation by exactly `d`, so the
        displacement needed to push a star at separation `rho` outside the
        annulus is `outer_radius - rho + margin`.

        Candidates are returned sorted by increasing distance, so the first
        one that verifies clean is also the smallest sufficient move.

        Parameters
        ----------
        stars : `list` of `dict`
            The `hit_info` entries for a single compromised pointing.

        Returns
        -------
        candidates : `list` of `tuple`
            `(dist_rad, bearing_rad)` pairs, within the offset budget.
        """
        candidates = []

        if len(stars) > 0:
            seps = np.array([s["sep_rad"] for s in stars])
            bearings = np.array([s["bearing_rad"] for s in stars])
            mags = np.array([s["star_mag_band"] for s in stars])

            # Distance needed to eject each star past the outer radius.
            needed = self.outer_radius_rad - seps + self.escape_margin_rad

            # 1. Primary push: directly away from the brightest offender.
            i_bright = int(np.argmin(mags))
            candidates.append((float(needed[i_bright]), float(bearings[i_bright] + np.pi)))

            # 2. Per-star pushes, in case the brightest star is already near
            #    the outer edge while another sits mid-annulus.
            for k in range(len(stars)):
                if k != i_bright:
                    candidates.append((float(needed[k]), float(bearings[k] + np.pi)))

            # 3. Consensus push: away from the mean direction of all
            #    offenders, far enough to clear the worst of them. Handles the
            #    common case of two stars on roughly the same side.
            if len(stars) > 1:
                mean_x = np.mean(np.cos(bearings))
                mean_y = np.mean(np.sin(bearings))
                if np.hypot(mean_x, mean_y) > 1e-9:
                    consensus = np.arctan2(mean_y, mean_x) + np.pi
                    candidates.append((float(np.max(needed)), float(consensus)))

        # 4. Ring fallback: sweep bearings at a few radii. Covers geometries
        #    where offenders straddle the field and no push-away direction
        #    works on its own.
        if self.n_ring_bearings > 0 and self.max_offset_rad > 0:
            ring_bearings = np.linspace(0.0, 2.0 * np.pi, self.n_ring_bearings, endpoint=False)
            for frac in (0.5, 0.75, 1.0):
                dist = frac * self.max_offset_rad
                for b in ring_bearings:
                    candidates.append((float(dist), float(b)))

        # Keep only physically useful moves inside the budget, cheapest first.
        candidates = [(d, b) for d, b in candidates if 0.0 < d <= self.max_offset_rad]
        candidates.sort(key=lambda c: c[0])

        return candidates

    def _adaptive_offset(self, orig_ra, orig_dec, obs_band, stars):
        """Find the smallest great-circle move that clears the annulus.

        Candidate displacements are generated by `_candidate_offsets` and then
        verified in a single batched `check_pointings` call. Verification is
        what guarantees correctness: a candidate is accepted only if the moved
        pointing has no bright star in the annulus, which also rules out moves
        that drag a different star in from the far side.

        A candidate must additionally be an *outward* escape -- every original
        offender ends up at or beyond `outer_radius_deg`. `check_pointings`
        alone would also accept moves that pull a star inside the inner radius,
        toward the science field of view, which is not acceptable here.

        Parameters
        ----------
        orig_ra, orig_dec : `float`
            Original pointing center in radians.
        obs_band : `str`
            Observation band, used for the magnitude threshold.
        stars : `list` of `dict`
            The `hit_info` entries for this pointing.

        Returns
        -------
        new_ra, new_dec : `float`
            Updated pointing center in radians (wrapped).
        info : `dict`
            Diagnostics: realized distance/bearing, whether the annulus was
            successfully cleared, and how many candidates were evaluated.
        """
        candidates = self._candidate_offsets(stars)

        if len(candidates) == 0:
            return (
                orig_ra,
                orig_dec,
                {
                    "offset_dist_deg": 0.0,
                    "offset_bearing_deg": float("nan"),
                    "escaped": False,
                    "n_candidates_tried": 0,
                },
            )

        dists = np.array([c[0] for c in candidates])
        bearings = np.array([c[1] for c in candidates])

        # dest_latlon returns (lat, lon) == (dec, ra) -- note the order.
        trial_dec, trial_ra = dest_latlon(dists, bearings, orig_dec, orig_ra)
        trial_ra, trial_dec = wrap_ra_dec(
            np.asarray(trial_ra, dtype=float).copy(),
            np.asarray(trial_dec, dtype=float).copy(),
        )

        compromised, hit_info = self.check_pointings(trial_ra, trial_dec, bands=[obs_band] * len(candidates))

        # Enforce outward-only escapes. `check_pointings` reports a pointing as
        # clean whenever no bright star sits in the annulus, which includes
        # moves that pull an offender *inside* the inner radius -- toward the
        # science field of view. Require every original offender to end up at
        # or beyond the outer radius instead.
        outward = np.ones(len(candidates), dtype=bool)
        if len(stars) > 0:
            star_idx = np.array([s["star_idx"] for s in stars])
            for k in range(len(candidates)):
                seps_after = _angular_separation(
                    trial_ra[k],
                    trial_dec[k],
                    self.star_ra_rad[star_idx],
                    self.star_dec_rad[star_idx],
                )
                if np.any(np.atleast_1d(seps_after) < self.outer_radius_rad):
                    outward[k] = False

        clean = np.where(~compromised & outward)[0]
        if len(clean) > 0:
            # Candidates are distance-sorted, so the first clean one is the
            # smallest sufficient move.
            pick = int(clean[0])
            escaped = True
        else:
            # Best effort: minimize the remaining in-annulus flux.
            flux = np.array(
                [
                    np.sum([10.0 ** (-0.4 * s["star_mag_band"]) for s in info]) if len(info) > 0 else 0.0
                    for info in hit_info
                ]
            )
            pick = int(np.argmin(flux))
            escaped = False

        info = {
            "offset_dist_deg": float(np.degrees(dists[pick])),
            "offset_bearing_deg": float(np.degrees(bearings[pick]) % 360.0),
            "escaped": escaped,
            "n_candidates_tried": len(candidates),
        }

        return float(trial_ra[pick]), float(trial_dec[pick]), info

    def __call__(self, observation_array, conditions=None):
        """Process and detail candidate observations.

        Parameters
        ----------
        observation_array : `ObservationArray`
            The candidate observations to detail.
        conditions : `rubin_scheduler.scheduler.conditions` object, optional
            Current environmental conditions from the scheduler.

        Returns
        -------
        observation_array : `ObservationArray`
            The updated observation array, with coordinates shifted for any
            compromised pointings.
        """
        if len(observation_array) == 0:
            return observation_array

        self.n_evaluated += len(observation_array)
        ra = observation_array["RA"]
        dec = observation_array["dec"]

        bands = observation_array["band"] if "band" in observation_array.dtype.names else None

        compromised, hit_info = self.check_pointings(ra, dec, bands=bands)

        if not np.any(compromised):
            return observation_array

        comp_indices = np.where(compromised)[0]
        self.n_shifted += len(comp_indices)

        # Create a copy to avoid modifying the original survey state
        observation_array = observation_array.copy()

        for idx in comp_indices:
            orig_ra = float(ra[idx])
            orig_dec = float(dec[idx])
            obs_band = (
                str(observation_array["band"][idx]).lower()
                if "band" in observation_array.dtype.names
                else "v"
            )

            if obs_band in self.n_shifted_by_band:
                self.n_shifted_by_band[obs_band] += 1
            else:
                self.n_shifted_by_band["other"] += 1

            new_ra, new_dec, offset_info = self._adaptive_offset(orig_ra, orig_dec, obs_band, hit_info[idx])
            if not offset_info["escaped"]:
                self.n_failed += 1

            wrapped_ra, wrapped_dec = wrap_ra_dec(np.array([new_ra]), np.array([new_dec]))
            observation_array["RA"][idx] = wrapped_ra[0]
            observation_array["dec"][idx] = wrapped_dec[0]

            # Annotate scheduler notes to trace which visits were modified
            tag = self.note_tag if offset_info["escaped"] else f"{self.note_tag}_fail"
            if self.annotate_note:
                for col in ["observation_reason"]:
                    if col in observation_array.dtype.names:
                        current = str(observation_array[col][idx])
                        if tag not in current:
                            max_len = 59
                            separator = ", " if current else ""
                            combined = f"{current}{separator}{tag}"[:max_len]
                            observation_array[col][idx] = combined

            # Record shift history
            # Realized per-axis offsets, recorded for analysis alongside the
            # great-circle distance/bearing that actually produced the move.
            realized_ra_deg = float(np.degrees(wrapped_ra[0] - orig_ra))
            # Unwrap the RA difference into (-180, 180].
            realized_ra_deg = (realized_ra_deg + 180.0) % 360.0 - 180.0
            realized_dec_deg = float(np.degrees(wrapped_dec[0] - orig_dec))

            if self.log_offsets:
                self.shifted_history.append(
                    {
                        "orig_ra_deg": float(np.degrees(orig_ra)),
                        "orig_dec_deg": float(np.degrees(orig_dec)),
                        "new_ra_deg": float(np.degrees(wrapped_ra[0])),
                        "new_dec_deg": float(np.degrees(wrapped_dec[0])),
                        "offset_ra_deg": realized_ra_deg,
                        "offset_dec_deg": realized_dec_deg,
                        "offset_dist_deg": offset_info["offset_dist_deg"],
                        "offset_bearing_deg": offset_info["offset_bearing_deg"],
                        "escaped": offset_info["escaped"],
                        "n_candidates_tried": offset_info["n_candidates_tried"],
                        "band": obs_band,
                        "stars": hit_info[idx],
                        "night": getattr(conditions, "night", None) if conditions else None,
                        "mjd": (
                            float(observation_array["mjd"][idx])
                            if "mjd" in observation_array.dtype.names
                            else None
                        ),
                    }
                )

        return observation_array

    def summary(self):
        """Print a summary of the detailer's actions."""
        pct = (self.n_shifted / self.n_evaluated * 100.0) if self.n_evaluated > 0 else 0.0
        print("=" * 65)
        print("HorseshoeAvoidanceDetailer Summary:")
        print(f"  HEALPix NSIDE:      {self.nside}")
        print(f"  Catalog:            {self.catalog_path.name}")
        print(f"  Stars indexed:      {len(self.xyz_stars)}")
        print(
            f"  Evaluation Mode:    {'Per-Observation Filter' if self.use_observation_band else 'V-band Only'}"
        )
        if isinstance(self.mag_limit, dict):
            print(f"  Magnitude Limits:   {self.mag_limit}")
        else:
            print(f"  Magnitude Limit:    Mag <= {self.mag_limit} in reference filter")
        print(f"  Annulus:            [{self.inner_radius_deg:.3f}, {self.outer_radius_deg:.3f}] deg")
        print("  Offset mode:        adaptive (push away from star)")
        print(f"  Offset budget:      <= {self.max_offset_deg:.4f} deg")
        print(f"  Escape margin:      {self.escape_margin_deg:.4f} deg")
        dists = [
            h["offset_dist_deg"]
            for h in self.shifted_history
            if np.isfinite(h.get("offset_dist_deg", np.nan))
        ]
        if dists:
            print(
                f"  Offset applied:     median {np.median(dists):.4f}, "
                f"95th {np.percentile(dists, 95):.4f}, "
                f"max {np.max(dists):.4f} deg"
            )
        print(f"  Evaluated visits:   {self.n_evaluated}")
        print(f"  Shifted visits:     {self.n_shifted} ({pct:.3f}%)")
        fail_pct = (self.n_failed / self.n_shifted * 100.0) if self.n_shifted > 0 else 0.0
        print(f"  Failed to escape:   {self.n_failed} ({fail_pct:.3f}% of shifted)")
        print("  Shifted by band:")
        for b in ["u", "g", "r", "i", "z", "y"]:
            cnt = self.n_shifted_by_band.get(b, 0)
            print(f"    - Band {b}: {cnt:4d} visits")
        print("=" * 65)

    def get_shifted_dataframe(self):
        """Return history of shifted visits as a pandas DataFrame."""
        if not self.shifted_history:
            return pd.DataFrame()
        return pd.DataFrame(self.shifted_history)
