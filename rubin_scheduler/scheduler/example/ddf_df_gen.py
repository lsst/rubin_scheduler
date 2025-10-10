__all__ = ("generate_ddf_df",)

import copy

import pandas as pd


def generate_ddf_df() -> pd.DataFrame:
    """Define the sequences for each field"""

    short_squences = [
        {
            "u": 3,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 33,
        },
        {
            "y": 2,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 33,
        },
        {
            "g": 2,
            "i": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 56,
            "even_odd": "even",
        },
        {
            "r": 2,
            "z": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 56,
            "even_odd": "odd",
        },
    ]

    shallow_squences = [
        {
            "u": 3,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "y": 2,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "g": 2,
            "i": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 100,
            "even_odd": "even",
        },
        {
            "r": 2,
            "z": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 100,
            "even_odd": "odd",
        },
    ]

    deep_sequences = [
        {
            "u": 8,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "y": 20,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "g": 2,
            "i": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 200,
            "even_odd": "even",
        },
        {
            "r": 2,
            "z": 2,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 200,
            "even_odd": "odd",
        },
        {
            "g": 4,
            "r": 18,
            "i": 55,
            "z": 52,
            "season_length": 180,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 110,
        },
    ]

    euclid_deep_seq = [
        {
            "u": 30,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "y": 40,
            "season_length": 225,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
        {
            "g": 4,
            "i": 4,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 200,
            "even_odd": "even",
        },
        {
            "r": 4,
            "z": 4,
            "season_length": 225,
            "flush_length": 0.5,
            "g_depth_limit": 22.8,
            "n_sequences": 200,
            "even_odd": "odd",
        },
        {
            "g": 8,
            "r": 36,
            "i": 110,
            "z": 104,
            "season_length": 125,
            "flush_length": 2.0,
            "g_depth_limit": 23.5,
            "n_sequences": 75,
        },
    ]

    short_seasons = {
        "XMM_LSS": [0, 10],
        "ELAISS1": [0, 10],
        "ECDFS": [0, 10],
        "EDFS_a": [0, 10],
    }

    shallow_seasons = {
        "COSMOS": [0, 4, 5, 6, 7, 8, 9, 10],
        "XMM_LSS": [1, 2, 3, 5, 6, 7, 8, 9],
        "ELAISS1": [1, 2, 3, 4, 6, 7, 8, 9],
        "ECDFS": [1, 2, 3, 4, 5, 7, 8, 9],
        "EDFS_a": [2, 3, 4, 5, 6, 7, 8, 9],
    }

    deep_seasons = {
        "COSMOS": [1, 2, 3],
        "XMM_LSS": [4],
        "ELAISS1": [5],
        "ECDFS": [6],
        "EDFS_a": [1],
    }

    dataframes = []

    for ddf_name in short_seasons:
        for season in short_seasons[ddf_name]:
            dict_for_df = {
                "ddf_name": ddf_name,
                "season": season,
                "even_odd": "None",
            }
            for key in "ugrizy":
                dict_for_df[key] = 0

            for seq in short_squences:
                row = copy.copy(dict_for_df)
                for key in seq:
                    row[key] = seq[key]
                dataframes.append(pd.DataFrame.from_dict(row, orient="index").T)

    for ddf_name in shallow_seasons:
        for season in shallow_seasons[ddf_name]:
            dict_for_df = {
                "ddf_name": ddf_name,
                "season": season,
                "even_odd": "None",
            }
            for key in "ugrizy":
                dict_for_df[key] = 0

            for seq in shallow_squences:
                row = copy.copy(dict_for_df)
                for key in seq:
                    row[key] = seq[key]
                dataframes.append(pd.DataFrame.from_dict(row, orient="index").T)

    for ddf_name in deep_seasons:
        for season in deep_seasons[ddf_name]:
            dict_for_df = {
                "ddf_name": ddf_name,
                "season": season,
                "even_odd": "None",
            }
            for key in "ugrizy":
                dict_for_df[key] = 0
            if ddf_name == "EDFS_a":
                for seq in euclid_deep_seq:
                    row = copy.copy(dict_for_df)
                    for key in seq:
                        row[key] = seq[key]
                    dataframes.append(pd.DataFrame.from_dict(row, orient="index").T)
            else:
                for seq in deep_sequences:
                    row = copy.copy(dict_for_df)
                    for key in seq:
                        row[key] = seq[key]

                    dataframes.append(pd.DataFrame.from_dict(row, orient="index").T)

    result = pd.concat(dataframes)

    return result
