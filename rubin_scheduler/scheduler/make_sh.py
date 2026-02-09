import glob


if __name__ == "__main__":

    filenames = glob.glob("*.npz")
    nights_needed = []

    for i in range(365):

        if "sat_streak_results_%i.npz" % i not in filenames:
            nights_needed.append(i)

    for night in nights_needed:
        print("sat_streak_results_%i.npz" % night)

