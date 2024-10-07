__all__ = ("gen_all_events",)

import os
import sqlite3

import healpy as hp
import numpy as np
import pandas as pd

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.utils import SimTargetooServer, TargetoO
from rubin_scheduler.utils import _angular_separation, _hpid2_ra_dec


def gen_gw_events(mjd_start=61557, mjd_end=62502, scale=1, seed=42):
    """Generate gravitational wave events"""

    """
    Case A) 9 triggers with 30 sq deg area -> should yield 3 visible.
    Case B) 15 triggers with 50 sq deg area -> should yield 5 observable.
    Case C) 24 triggers with 100 sq deg area -> should yield 8 observable.
    Case D) 6 triggers with 250 sq deg area -> should yield 4 observable.
    Case E) 12 triggers with 500 sq deg area -> should yield 2 observable.
    """

    rng = np.random.default_rng(seed=seed)

    case_2_area = {"A": 30.0, "B": 50, "C": 100, "D": 250, "E": 500}
    case_2_nt = {"A": 9, "B": 15, "C": 24, "D": 6, "E": 12}

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


def gen_bbh_events(mjd_start=61557, mjd_end=62502, scale=1, seed=43):
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
    n_trigger=160.0,
    mjd_start=60796,
    mjd_end=60796 + 3652.5,
    radius=0.0,
):
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


def gen_lensed_BNS(mjd_start=61557, mjd_end=62502, scale=1, seed=45):

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


def gen_sso_events(n_events=300, twi_fraction=0.75, seed=52, radius=2.0):
    """Maybe just read in baseline to be sure we select things
    that would be visible
    """

    rng = np.random.default_rng(seed=seed)
    data_dir = get_data_dir()
    con = sqlite3.connect(os.path.join(data_dir, "no_too.db"))
    df = pd.read_sql(
        ('select fieldRA as ra,fieldDec as dec, observationStartMJD'
         'from observations where scheduler_note like "twilight%"'),
        con,
    )
    names = ["mjd_start", "ra", "dec", "expires", "radius", "ToO_label"]
    types = [float] * 5 + ["<U50"]
    twi_events = np.zeros(int(n_events * twi_fraction), dtype=list(zip(names, types)))

    indx = rng.choice(np.arange(df["ra"].values.size), size=twi_events.size)
    twi_events["ra"] = np.radians(df["ra"].values[indx])
    twi_events["dec"] = np.radians(df["dec"].values[indx])
    twi_events["mjd_start"] = df["observationStartMJD"].values[indx] + 0.65

    twi_events["ToO_label"] = "SSO_twilight"

    df = pd.read_sql(
        ('select fieldRA as ra,fieldDec as dec,observationStartMJD'
         'from observations where scheduler_note not like "twilight%"'),
        con,
    )

    reg_events = np.zeros(int(n_events - twi_events.size), dtype=list(zip(names, types)))
    indx = rng.choice(np.arange(df["ra"].values.size), size=reg_events.size)
    reg_events["ra"] = np.radians(df["ra"].values[indx])
    reg_events["dec"] = np.radians(df["dec"].values[indx])
    reg_events["mjd_start"] = df["observationStartMJD"].values[indx] + 0.65

    reg_events["ToO_label"] = "SSO_night"

    events = np.concatenate([twi_events, reg_events])
    events["expires"] = events["mjd_start"] + 3
    events["radius"] = np.radians(radius)

    return events


def gen_all_events(scale=1, nside=32, include_gw=True, include_neutrino=True, include_ss=True):

    if scale == 0:
        return None, None

    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

    events = []
    if include_gw:
        events.append(gen_gw_events(scale=3 * scale))
        events.append(gen_bbh_events(scale=3 * scale))
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
