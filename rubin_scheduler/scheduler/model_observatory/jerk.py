__all__ = ("jerk_time", "acc_time")

import numpy as np


def jerk_time(distance, v_max, acc_max, jerk_max):
    """Calculate how long to move a distance given maximum jerk,
    acceleration, and velocity

    All parameters are assumed to be symetric
    (e.g, minimum jerk = -1*jerk_max)

    modified from https://github.com/mdhom/py_constant_jerk/blob/main/
    constantJerk.py
    see also:
    https://www.researchgate.net/publication/
    289374755_THIRD_ORDER_POINT-TO-POINT_MOTION_-PROFILE

    Parameters
    ----------
    distance : `float`
        The distance to travel. Could be units of length or angle as long
        as other parameters match.
    v_max : `float`
        The maximum velocity. Units of length or angle per unit of time
    acc_max : `float`
        The maximum acceleration. Units of length or angle per unit of
        time squared.
    jerk_max : `float`
        The maximum jerk. Units of length or angle per unit of time cubed.
        If set to None, assumes jerk is infinite and defaults to use
        acc_time.
    """

    # If distance is a scalar, convert to array.
    distance = np.atleast_1d(distance)

    # If there is no jerk, fall back on infinite jerk approximation
    if jerk_max is None:
        return acc_time(distance, v_max, acc_max)

    trajectory_instance_case = _get_trajectory_instance_case(distance, v_max, acc_max, jerk_max)

    unique_trag = np.unique(trajectory_instance_case)

    if np.size(unique_trag) == 1:
        tj, ta, tv = _calculate_times(
            distance,
            v_max,
            acc_max,
            jerk_max,
            trajectory_instance_case=unique_trag,
        )
        result = tv + tj + ta
    else:
        result = np.zeros(np.size(distance))
        for tic in unique_trag:
            indx = np.where(trajectory_instance_case == tic)[0]
            tj, ta, tv = _calculate_times(
                distance[indx],
                v_max,
                acc_max,
                jerk_max,
                trajectory_instance_case=tic,
            )
            result[indx] = tv + tj + ta

    return result


def acc_time(distance, v_max, acc_max):
    """Time to move given a maximum velocity and acceleration. Assumes
    infinite jerk. Velocity and accelerations are assumed to be
    symetric (minimum acceleration = -1* acc_max)

    Parameters
    ----------
    distance : `float`
        The distance to travel. Could be units of length or angle as long
        as other parameters match.
    v_max : `float`
        The maximum velocity. Units of length or angle per unit of time
    acc_max : `float`
        The maximum acceleration. Units of length or angle per unit of
        time squared.
    """

    distance = np.atleast_1d(distance)

    dm = v_max**2 / acc_max
    slew_time = np.where(
        distance < dm,
        2 * np.sqrt(distance / acc_max),
        2 * v_max / acc_max + (distance - dm) / v_max,
    )
    return slew_time


def _calculate_times(distance, v_max, acc_max, jerk_max, trajectory_instance_case=1):
    """Calculate travel time including jerk.

    Modified from https://github.com/mdhom/py_constant_jerk/blob/main/
    #constantJerk.py
    """
    if trajectory_instance_case == 1 or trajectory_instance_case == 3:
        tj = np.sqrt(v_max / jerk_max)
        ta = tj
        tv = distance / v_max
        return tj, ta, tv
    elif trajectory_instance_case == 2 or trajectory_instance_case == 4:
        tj = (distance / (2 * jerk_max)) ** (1.0 / 3.0)
        ta = tj
        tv = 2 * tj
        return tj, ta, tv
    elif trajectory_instance_case == 5:
        tj = acc_max / jerk_max
        ta = v_max / acc_max
        tv = distance / v_max
        return tj, ta, tv
    elif trajectory_instance_case == 6:
        tj = acc_max / jerk_max
        ta = 0.5 * (
            np.sqrt(
                (4 * distance * jerk_max * jerk_max + acc_max * acc_max * acc_max)
                / (acc_max * jerk_max * jerk_max)
            )
            - acc_max / jerk_max
        )
        tv = ta + tj
        return tj, ta, tv
    else:
        raise Exception("TrajectoryInstance must be between 1 and 6")


def _get_trajectory_instance_case(distance, v_max, acc_max, jerk_max):
    """Determine which motion profile to use.

    Modified from https://github.com/mdhom/py_constant_jerk/blob/main/
    constantJerk.py
    which was probably derived from:
    https://www.researchgate.net/publication/
    289374755_THIRD_ORDER_POINT-TO-POINT_MOTION_-PROFILE

    Parameters
    ----------
    distance : `float`
        The distance to travel. Could be units of length or angle as long
        as other parameters match.
    v_max : `float`
        The maximum velocity. Units of length or angle per unit of time.
    acc_max : `float`
        The maximum acceleration. Units of length or angle per unit of
        time squared.
    jerk_max : `float`
        The maximum jerk. Units of length or angle per unit of time cubed.

    """
    # Case 1: a_max reached, v_max reached and constant for a time
    #          (s = 100,   j = 2000, a = 500,  vMax = 120)
    # Case 2: a_max not reached, v_max not reached
    #          (s = 15000, j = 2000, a = 5500, vMax = 20500)
    # Case 3: a_max not reached, v_max reached and constant for a time
    #          (s = 15000, j = 2000, a = 5500, vMax = 2500)
    # Case 4: a_max reached, v_max not reached
    #          (s = 57,    j = 2000, a = 500,  vMax = 120)
    # Case 5: a_max reached and constant for a time, v_max reached and
    # constant for a time
    #          (s = 15000, j = 2000, a = 500,  vMax = 2500)
    # Case 6: a_max reached and constant for a time, v_max not reached
    #          (s = 15000, j = 2000, a = 500,  vMax = 20500)

    result = np.array(distance * 0)

    v_a = acc_max * acc_max / jerk_max
    s_a = 2 * acc_max * acc_max * acc_max / (jerk_max * jerk_max)
    s_v = 0.0
    if v_max * jerk_max < acc_max * acc_max:
        s_v = v_max * 2 * np.sqrt(v_max / jerk_max)
    else:
        s_v = v_max * (v_max / acc_max + acc_max / jerk_max)

    result[np.where((v_max < v_a) & (distance >= s_a))] = 1
    result[np.where((v_max >= v_a) & (distance < s_a))] = 2
    result[np.where((v_max < v_a) & (s_a > distance) & (distance >= s_v))] = 3
    result[np.where((v_max < v_a) & (distance < s_a) & (distance < s_v))] = 4
    result[np.where((v_max >= v_a) & (distance >= s_a) & (distance >= s_v))] = 5
    result[np.where((v_max >= v_a) & (s_a <= distance) & (distance < s_v))] = 6

    return result
