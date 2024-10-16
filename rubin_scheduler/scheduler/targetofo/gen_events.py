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
O5_START_MJD = 61557
O5_END_MJD = 62502


def gen_gw_events(mjd_start=O5_START_MJD, mjd_end=O5_END_MJD, scale=1, seed=42):
    """Generate gravitational wave events

    Parameters
    ----------
    mjd_start : `float`
        MJD to start creating events. Default O5_START_MJD set to
        expected O5 run from GW detectors.
    mjd_end : `float`
        MJD to stop creating events. Default O5_END_MJD set to
        expected end of O5 run from GW detectors.
    scale : `float`
        How much to scale the expected number of evenets.
        Expect 22 events if set to default of 1.
    seed : `float`
        Random number generator seed. Default 42.
    """

    """
    Case A) 9 triggers with 30 sq deg area -> should yield 3 visible.
    Case B) 15 triggers with 50 sq deg area -> should yield 5 observable.
    Case C) 24 triggers with 100 sq deg area -> should yield 8 observable.
    Case D) 6 triggers with 250 sq deg area -> should yield 4 observable.
    Case E) 12 triggers with 500 sq deg area -> should yield 2 observable.
    """
    case_2_area = {"A": 30.0, "B": 50, "C": 100, "D": 250, "E": 500}
    case_2_nt = {"A": 9, "B": 15, "C": 24, "D": 6, "E": 12}

    rng = np.random.default_rng(seed=seed)

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
        event_table["ToO_label"] = "GW_case_%s" % case
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


def gen_bbh_events(mjd_start=O5_START_MJD, mjd_end=O5_END_MJD, scale=1, seed=43):
    """Generate black hole - black home merger events.

    Parameters
    ----------
    mjd_start : `float`
        MJD to start creating events. Default O5_START_MJD set to
        expected O5 run from GW detectors.
    mjd_end : `float`
        MJD to stop creating events. Default O5_END_MJD set to
        expected end of O5 run from GW detectors.
    scale : `float`
        How much to scale the expected number of evenets.
        Expect 21 events if set to default of 1.
    seed : `float`
        Random number generator seed. Default 43.
    """

    rng = np.random.default_rng(seed=seed)
    """
    Case A) 9 triggers with 10 sq deg area -> yields 3 observable
    Case B) 6 triggers with 20 sq deg area -> yields 2 observable
    Case C) 6 triggers with 30 sq deg area -> yields 2 observable
    """
    case_2_area = {"A": 10.0, "B": 20, "C": 30}
    case_2_nt = {"A": 9, "B": 6, "C": 6}

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
        event_table["ToO_label"] = "BBH_case_%s" % case
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


def gen_sso_events(n_events=900, twi_fraction=0.75, seed=52, radius=2.0, mjd_start=SURVEY_START_MJD):
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
    potential_mjds = np.arange(0, 3652, 4) + mjd_start + 1
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

    potential_mjds = np.arange(0, 3652) + mjd_start
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


def gen_all_events(scale=1, nside=DEFAULT_NSIDE, include_gw=True, include_neutrino=True, include_ss=True):
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
        events.append(gen_neutrino_events())
    # Not varying the number of SSO events
    if include_ss:
        events.append(gen_sso_events())
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
