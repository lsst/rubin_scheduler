__all__ = ("gen_all_events",)


import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time

from rubin_scheduler.scheduler.utils import SimTargetooServer, TargetoO
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD, _angular_separation, _hpid2_ra_dec

# Expected start and end for observing run 5
# See: https://observing.docs.ligo.org/plan/
O5_START_MJD = 62137
O5_END_MJD = 63110

# Expected start and end for intermediate run 1
# See: https://observing.docs.ligo.org/plan/
IR1_START_MJD = 61423
IR1_END_MJD = 61603

# Annual all-sky public-alert rates (median) for compact binary mergers.
# https://rtd.igwn.org/projects/userguide/en/stable/capabilities.html
# IR1 uses the O4 HLV row; O5a/b/c are the three O5 sensitivity scenarios.
GW_ALERT_RATES = {
    "IR1": {"BNS": 2, "NSBH": 4, "BBH": 240},
    "O5a": {"BNS": 5, "NSBH": 11, "BBH": 520},
    "O5b": {"BNS": 10, "NSBH": 21, "BBH": 760},
    "O5c": {"BNS": 15, "NSBH": 25, "BBH": 890},
}


# (Gold, Silver) fraction of alerts; Bronze is the remainder. Read from the
# cumulative-rate curves in
# https://rtd.igwn.org/projects/userguide/en/stable/_images/annual-predictions.svg
# at 90% area of 100 and 500 deg2. The Bronze remainder is every alert above
# 500 deg2, which the ToO report trims to 1500 deg2.
GW_NS_TIER_FRACTIONS = {
    "IR1": {"BNS": (0.069, 0.165), "NSBH": (0.030, 0.117)},
    "O5a": {"BNS": (0.115, 0.187), "NSBH": (0.143, 0.151)},
    "O5b": {"BNS": (0.073, 0.173), "NSBH": (0.079, 0.127)},
    "O5c": {"BNS": (0.103, 0.199), "NSBH": (0.107, 0.161)},
}

# Localization area bounds per tier (deg2); events are drawn uniformly in area.
GW_TIER_AREA_BOUNDS = {"Gold": (10.0, 100.0), "Silver": (100.0, 500.0), "Bronze": (500.0, 1500.0)}

# Fraction of BBH alerts with full 90% area < 100 deg2, read from the BBH
# cumulative-rate curves in annual-predictions.svg.
GW_BBH_AREA_FRACTION = {"IR1": 0.067, "O5a": 0.138, "O5b": 0.108, "O5c": 0.138}

# Fraction of those with total mass > 50 Msun (chirp mass >= 22 Msun). From the
# ToO report (Sec. 2.2.3): ~8 events/yr in O4 for area<100 deg2 and M>50, vs
# ~16/yr below 100 deg2 at all masses from the O4-HLV curve.
GW_BBH_HIGH_MASS_FRACTION = 0.50

# Report trigger cut is a full-skymap 90% area < 100 deg2 (<= 20 pointings).
GW_BBH_AREA_BOUNDS = (10.0, 100.0)


def _gw_epochs(mjd_start, mjd_end):
    """Yield (epoch, t0, t1, years) for IR1 and O5a/b/c, clipped to
    [mjd_start, mjd_end]. O5a/b/c are sequential thirds of O5.
    """
    o5_edges = np.linspace(O5_START_MJD, O5_END_MJD, 4)
    epochs = {
        "IR1": (IR1_START_MJD, IR1_END_MJD),
        "O5a": (o5_edges[0], o5_edges[1]),
        "O5b": (o5_edges[1], o5_edges[2]),
        "O5c": (o5_edges[2], o5_edges[3]),
    }
    for epoch, (t0, t1) in epochs.items():
        t0, t1 = max(t0, mjd_start), min(t1, mjd_end)
        if t1 > t0:
            yield epoch, t0, t1, (t1 - t0) / 365.25


def gen_gw_events(mjd_start=IR1_START_MJD, mjd_end=O5_END_MJD, scale=1, seed=42):
    """Generate gravitational wave events from neutron star-bearing mergers
    (BNS and NSBH), labeled by observing-strategy tier.

    Counts are Poisson draws around the expected IGWN public-alert rate for
    each epoch (IR1, then O5a/b/c as sequential thirds of O5), split into
    Gold/Silver/Bronze by 90% localization area (<100, <500, >500 deg2).
    Events are all-sky; visibility is applied by the simulation.

    Parameters
    ----------
    mjd_start : `float`
        MJD to start creating events. Default IR1_START_MJD.
    mjd_end : `float`
        MJD to stop creating events. Default O5_END_MJD. Epochs are
        clipped to [mjd_start, mjd_end].
    scale : `float`
        Multiplier on the expected number of events. Default 1.
    seed : `float`
        Random number generator seed. Default 42.
    """
    rng = np.random.default_rng(seed=seed)

    names = ["mjd_start", "ra", "dec", "expires", "radius", "ToO_label"]
    types = [float] * 5 + ["<U50"]
    dtype = list(zip(names, types))

    event_tables = [np.zeros(0, dtype=dtype)]
    for epoch, t0, t1, years in _gw_epochs(mjd_start, mjd_end):
        for source in ("BNS", "NSBH"):
            gold, silver = GW_NS_TIER_FRACTIONS[epoch][source]
            fractions = {"Gold": gold, "Silver": silver, "Bronze": 1.0 - gold - silver}
            for tier, frac in fractions.items():
                n = rng.poisson(GW_ALERT_RATES[epoch][source] * frac * years * scale)
                event_table = np.zeros(n, dtype=dtype)
                area = rng.uniform(*GW_TIER_AREA_BOUNDS[tier], size=n)
                event_table["radius"] = np.radians((area / np.pi) ** 0.5)
                event_table["mjd_start"] = rng.uniform(t0, t1, size=n)
                event_table["ToO_label"] = "GW_%s_%s" % (source, tier)
                event_tables.append(event_table)

    events = np.concatenate(event_tables)
    n_events = np.size(events)

    # Make sure latitude points spread correctly
    # http://mathworld.wolfram.com/SpherePointPicking.html
    events["ra"] = rng.random(n_events) * np.pi * 2
    events["dec"] = np.arccos(2.0 * rng.random(n_events) - 1.0) - np.pi / 2.0
    events["expires"] = events["mjd_start"] + 3

    return events


def gen_bbh_events(mjd_start=IR1_START_MJD, mjd_end=O5_END_MJD, scale=1, seed=43):
    """Generate high-mass black hole - black hole merger events that pass
    the ToO report trigger cuts (90% area < 100 deg2, chirp mass >= 22 Msun).

    Counts are Poisson draws around the expected IGWN public-alert rate in
    each epoch (IR1, then O5a/b/c), times the fraction passing the area and
    mass cuts. Events are all-sky: the 1/3 of events inside the LSST
    footprint is applied by the survey's followup_footprint, not here.

    Parameters
    ----------
    mjd_start : `float`
        MJD to start creating events. Default IR1_START_MJD.
    mjd_end : `float`
        MJD to stop creating events. Default O5_END_MJD.
    scale : `float`
        Multiplier on the expected number of events. Default 1.
    seed : `float`
        Random number generator seed. Default 43.
    """
    rng = np.random.default_rng(seed=seed)

    names = ["mjd_start", "ra", "dec", "expires", "radius", "ToO_label"]
    types = [float] * 5 + ["<U50"]
    dtype = list(zip(names, types))

    event_tables = [np.zeros(0, dtype=dtype)]
    for epoch, t0, t1, years in _gw_epochs(mjd_start, mjd_end):
        mean = (
            GW_ALERT_RATES[epoch]["BBH"]
            * GW_BBH_AREA_FRACTION[epoch]
            * GW_BBH_HIGH_MASS_FRACTION
            * years
            * scale
        )
        n = rng.poisson(mean)
        event_table = np.zeros(n, dtype=dtype)
        area = rng.uniform(*GW_BBH_AREA_BOUNDS, size=n)
        event_table["radius"] = np.radians((area / np.pi) ** 0.5)
        event_table["mjd_start"] = rng.uniform(t0, t1, size=n)
        event_table["ToO_label"] = "BBH"
        event_tables.append(event_table)

    events = np.concatenate(event_tables)
    n_events = np.size(events)

    # Make sure latitude points spread correctly
    # http://mathworld.wolfram.com/SpherePointPicking.html
    events["ra"] = rng.random(n_events) * np.pi * 2
    events["dec"] = np.arccos(2.0 * rng.random(n_events) - 1.0) - np.pi / 2.0
    events["expires"] = events["mjd_start"] + 3

    return events


def gen_neutrino_events(
    scale=1,
    seed=44,
    n_trigger=160,
    mjd_start=SURVEY_START_MJD,
    mjd_end=SURVEY_START_MJD + 3652.5,
    radius=0.0,
):
    """Generate neutrino ToO events

    Parameters
    ----------
    scale : `float`
        How much to scale number of events by. Defualt 1.
    n_trigger : `int`
        Number of events to trigger. Default 160.
    mjd_start : `float`
        Starting MJD. Default SURVEY_START_MJD.
    mjd_end : `float`
        Ending MJD. Default SURVEY_START_MJD + 3652.5.
    radius : `float`
        The search radius of the events on the sky.
        Default 0 (degrees).
    """
    rng = np.random.default_rng(seed=seed)
    n_events = int(n_trigger * scale)

    names = ["mjd_start", "ra", "dec", "expires", "radius", "ToO_label"]
    types = [float] * 5 + ["<U50"]

    events = np.zeros(n_events, dtype=list(zip(names, types)))

    events["ra"] = rng.random(n_events) * np.pi * 2
    events["dec"] = np.arccos(2.0 * rng.random(n_events) - 1.0) - np.pi / 2.0
    events["mjd_start"] = rng.random(n_events) * (mjd_end - mjd_start) + mjd_start
    events["ToO_label"] = "neutrino"
    events["expires"] = events["mjd_start"] + 3
    events["radius"] = np.radians(radius)

    return events


def gen_lensed_BNS(mjd_start=O5_START_MJD, mjd_end=O5_END_MJD, scale=1, seed=45):
    """Generate lensed black hole - neutron star GW events.

    Parameters
    ----------
    scale : `float`
        How much to scale number of events by. Default 1.
        Expect 15 events with default scale.
    mjd_start : `float`
        Starting MJD. Default O5_START_MJD
    mjd_end : `float`
        Ending MJD. Defualt O5_END_MJD
    """

    rng = np.random.default_rng(seed=seed)

    """
    Case A) 9 triggers with 10 sq deg area -> yields 3 observable
    Case B) 6 triggers with 20 sq deg area -> yields 2 observable
    """

    case_2_area = {"A": 900, "B": 15}
    case_2_nt = {"A": 1, "B": 1}

    n_events = 0
    for case in case_2_nt:
        n_events += case_2_nt[case] * scale

    event_tables = []
    names = ["mjd_start", "ra", "dec", "expires", "radius", "ToO_label"]
    types = [float] * 5 + ["<U50"]

    for case in case_2_area:
        n = case_2_nt[case] * scale
        event_table = np.zeros(int(n), dtype=list(zip(names, types)))
        event_table["radius"] = (case_2_area[case] / np.pi) ** 0.5 * np.pi / 180.0
        event_table["ToO_label"] = "lensed_BNS_case_%s" % case
        event_tables.append(event_table)

    events = np.concatenate(event_tables)
    n_events = np.size(events)

    # Make sure latitude points spread correctly
    # http://mathworld.wolfram.com/SpherePointPicking.html
    events["ra"] = rng.random(n_events) * np.pi * 2
    events["dec"] = np.arccos(2.0 * rng.random(n_events) - 1.0) - np.pi / 2.0
    events["mjd_start"] = rng.random(n_events) * (mjd_end - mjd_start) + mjd_start
    events["expires"] = events["mjd_start"] + 3

    return events


def gen_sso_events(
    n_events=900,
    twi_fraction=0.75,
    seed=52,
    radius=2.0,
    mjd_start=SURVEY_START_MJD,
    mjd_end=SURVEY_START_MJD + 3652,
):
    """Generate solar system ToO events

    Originally set to 300 events, swapped to a more approximate
    generation of events and increased number of events to 900 to
    ensure enough trigger.

    Parameters
    ----------
    n_events : `int`
        The number of events to generate. Default 900
    twi_fraction : `float`
        Fraction of events to set as earth-interior alerts,
        e.g., events that are near the sun and only visible at
        twilight. Default 0.75.
    seed : `float`
        Random number seed. Default 52.
    radius : `float`
        Search radius on sky for events. Default 2 (degrees).
    mjd_start : `float`
        MJD of when to start generating events.
        Default SURVEY_START_MJD.
    """

    # Seems like solar elongations are around 43.3 degrees
    # with 2.05 degree standard deviation
    elong_mean = np.radians(43.3)
    elong_std = np.radians(2.05)
    pad = np.radians(0.5)

    n_nights = int(mjd_end - mjd_start)

    # Then ecliptic latitude between -40 and +15

    rng = np.random.default_rng(seed=seed)

    nside = 128
    ra_rad, dec_rad = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
    coord = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad, frame="icrs")
    eclip_lat_rad = coord.barycentrictrueecliptic.lat.rad

    near_eclip = np.where((np.radians(-40.0) < eclip_lat_rad) & (eclip_lat_rad < np.radians(15.0)))[0]

    eclip_mask = np.zeros(ra_rad.size, dtype=bool)
    eclip_mask[near_eclip] = True

    # Generate inner SS objects
    names = ["mjd_start", "ra", "dec", "expires", "radius", "ToO_label"]
    types = [float] * 5 + ["<U50"]
    # Array to hold events
    twi_events = np.zeros(int(n_events * twi_fraction), dtype=list(zip(names, types)))

    n_inner = int(n_events * twi_fraction)
    # We want to trigger on nights divisible by 4+1
    potential_mjds = np.arange(0, n_nights, 4) + mjd_start + 1
    twi_events["mjd_start"] = rng.choice(potential_mjds, size=n_inner, replace=False)

    times = Time(twi_events["mjd_start"], format="mjd")
    sun = get_sun(times)

    ras = []
    decs = []

    for i, time in enumerate(times):
        sun_dist = _angular_separation(sun[i].ra.rad, sun[i].dec.rad, ra_rad, dec_rad)
        elong_draw = rng.normal(loc=elong_mean, scale=elong_std, size=1)
        good = np.where(((elong_draw - pad) < sun_dist) & (sun_dist < (elong_draw + pad)) & eclip_mask)[0]

        ra, dec = _hpid2_ra_dec(nside, rng.choice(good, size=1))
        ras.append(ra[0])
        decs.append(dec[0])

    twi_events["ra"] = ras
    twi_events["dec"] = decs

    twi_events["ToO_label"] = "SSO_twilight"

    reg_events = np.zeros(int(n_events - twi_events.size), dtype=list(zip(names, types)))

    potential_mjds = np.arange(0, n_nights) + mjd_start
    reg_events["mjd_start"] = rng.choice(potential_mjds, size=reg_events.size, replace=False)
    times = Time(reg_events["mjd_start"], format="mjd")
    sun = get_sun(times)

    ras = []
    decs = []

    for i, time in enumerate(times):
        sun_dist = _angular_separation(sun[i].ra.rad, sun[i].dec.rad, ra_rad, dec_rad)
        good = np.where(
            (sun_dist < np.radians(160.0))
            & (sun_dist > np.radians(90.0))
            & (eclip_lat_rad > np.radians(-80))
            & (eclip_lat_rad < np.radians(25))
        )[0]
        ra, dec = _hpid2_ra_dec(nside, rng.choice(good, size=1))
        ras.append(ra[0])
        decs.append(dec[0])

    reg_events["ra"] = ras
    reg_events["dec"] = decs

    reg_events["ToO_label"] = "SSO_night"

    events = np.concatenate([twi_events, reg_events])
    events["expires"] = events["mjd_start"] + 3
    events["radius"] = np.radians(radius)

    return events


def gen_all_events(
    scale=1,
    nside=DEFAULT_NSIDE,
    include_gw=True,
    include_neutrino=True,
    include_ss=True,
    mjd_start=SURVEY_START_MJD,
    mjd_end=SURVEY_START_MJD + 3652.5,
):
    """Function to generate ToO events

    Parameters
    ----------
    scale : `float`
        Amount to scale the total number of events. Default 1.
    nside : `int`
        HEALpix nside. Default set to DEAFAULT_NSIDE.
    include_gw : `bool`
        Include gravitational wave events (GW, BH-BH, and lensed BH-NS events).
        Default True.
    include_neutrino : `bool`
        Include neutrino events. Default True.
    include_ss : `bool`
        Include solar system events. Default True.
    """

    if scale == 0:
        return None, None

    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

    events = []
    if include_gw:
        events.append(gen_gw_events(scale=3.0 * scale))
        events.append(gen_bbh_events(scale=3.0 * scale))
        events.append(gen_lensed_BNS(scale=3.0 * scale))
    # Not varying the number of neutrino events
    if include_neutrino:
        n_events = np.round(160 / 3652.5 * (mjd_end - mjd_start))
        events.append(gen_neutrino_events(mjd_start=mjd_start, mjd_end=mjd_end))
    # Not varying the number of SSO events
    if include_ss:
        n_events = np.round(900 / 3652.5 * (mjd_end - mjd_start))
        events.append(gen_sso_events(n_events=n_events, mjd_start=mjd_start, mjd_end=mjd_end))
    event_table = np.concatenate(events)

    event_table.sort(order="mjd_start")

    events = []
    for i, event_time in enumerate(event_table["mjd_start"]):
        dist = _angular_separation(ra, dec, event_table["ra"][i], event_table["dec"][i])
        good = np.where(dist <= event_table["radius"][i])
        footprint = np.zeros(ra.size, dtype=float)
        footprint[good] = 1
        events.append(
            TargetoO(
                i,
                footprint,
                event_time,
                event_table["expires"][i] - event_table["mjd_start"][i],
                ra_rad_center=event_table["ra"][i],
                dec_rad_center=event_table["dec"][i],
                too_type=event_table["ToO_label"][i],
            )
        )
    events = SimTargetooServer(events)

    return events, event_table
