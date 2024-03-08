__all__ = (
    "order_observations",
    "generate_dist_matrix",
    "route_length",
    "generate_hulls",
    "merge_hulls",
    "three_opt",
    "tsp_convex",
)

import itertools
from collections import deque

import numpy as np
import scipy.spatial as spatial

from .utils import IntRounded, gnomonic_project_toxy, mean_azimuth

# Solve Traveling Salesperson using convex hulls.
# re-write of https://github.com/jameskrysiak/ConvexSalesman/
# blob/master/convex_salesman.py
# This like a good explination too
# https://www.youtube.com/watch?v=syRSy1MFuho


def order_observations(lon, lat, optimize=False):
    """Use TSP solver to put observations in an order that minimizes
    angular distance traveled

    Parameters
    ----------
    lon : `float`
        A longitude-like (RA, azimuth) angle (radians).
    lat : `float`
        A latitude-like (dec, altitude) angle (radians).
    scale : `float`
        A factor to scale and round projections to force same
        machine precision cross-platforms (1e6).
    optimize : `bool`
        If the TSP should run extra optimization steps, default False
    """

    # Let's find a good spot to project the points to a plane
    mid_dec = (np.max(lat) - np.min(lat)) / 2.0 + np.min(lat)
    mid_ra = mean_azimuth(lon)
    # Project the coordinates to a plane. Could consider scaling
    # things to represent time between points rather than angular
    # distance.
    pointing_x, pointing_y = gnomonic_project_toxy(lon, lat, mid_ra, mid_dec)

    # Now I have a bunch of x,y pointings. Drop into TSP solver
    # to get an effiencent route
    towns = np.vstack((pointing_x, pointing_y)).T
    # Leaving optimize=False for speed. The optimization step doesn't
    # usually improve much.
    better_order = tsp_convex(towns, optimize=optimize)
    return better_order


def generate_dist_matrix(towns):
    """Generate the matrix for the distance between town i and j

    Parameters
    ----------
    towns : np.array
        The x,y positions of the towns
    """

    x = towns[:, 0]
    y = towns[:, 1]
    # Broadcast to i,j
    x_dist = x - x[:, np.newaxis]
    y_dist = y - y[:, np.newaxis]
    distances = np.sqrt(x_dist**2 + y_dist**2)

    return distances


def route_length(town_indx, dist_matrix):
    """Find the length of a route

    Parameters
    ----------
    town_indx : array of int
        The indices of the towns.
    dist_matrix : np.array
        The matrix where the (i,j) elements are the distance
        between the ith and jth town
    """

    # This closes the path and return to the start
    town_i = town_indx
    town_j = np.roll(town_indx, -1)
    distances = dist_matrix[town_i, town_j]
    return np.sum(distances)


def generate_hulls(towns):
    """Given an array of x,y points, sort them into concentric hulls

    Parameters
    ----------
    towns : np.array (n,2)
        Array of town x,y positions

    Returns
    -------
    list of lists of the indices of the concentric hulls
    """

    # The indices we have to sort
    all_indices = np.arange(towns.shape[0])
    # array to note if a town has been used in a hull
    indices_used = np.zeros(towns.shape[0], dtype=bool)
    results = []

    # Continue until every point is inside a convex hull.
    while False in indices_used:
        # Try to find the convex hull of the remaining points.
        try:
            new_hull = spatial.ConvexHull(towns[all_indices[~indices_used]])
            new_indices = all_indices[~indices_used][new_hull.vertices]
            results.append(new_indices.tolist())
            indices_used[new_indices] = True

        # In a degenerate case (fewer than three points, points collinear)
        # Add all of the remaining points to the innermost convex hull.
        except spatial._qhull.QhullError:
            results.append(all_indices[~indices_used].tolist())
            indices_used[~indices_used] = True
            return results

    return results


def merge_hulls(indices_lists, dist_matrix):
    """Combine the hulls

    Parameters
    ----------
    indices_list : list of lists with ints
    dist_matrix : np.array
    """
    # start with the outer hull one. Use deque to rotate fast.
    collapsed_indices = deque(indices_lists[0])
    for ind_list in indices_lists[1:]:
        # insert each point indvidually
        for indx in ind_list:
            possible_results = []
            possible_lengths = []
            dindex = deque([indx])
            # In theory, I think this could loop over fewer points.
            # Only need to check points that can "see" the inner points?
            for i in range(len(collapsed_indices)):
                collapsed_indices.rotate(1)
                possible_results.append(collapsed_indices + dindex)
                possible_lengths.append(route_length(possible_results[-1], dist_matrix))
            best = np.min(np.where(possible_lengths == np.nanmin(possible_lengths)))
            collapsed_indices = possible_results[best]
    return list(collapsed_indices)


def three_opt(route, dist_matrix, cross_platform=True):
    """Iterates over all possible 3-optional transformations.

    Parameters
    ----------
    route : `list`
        The indices of the route
    dist_matrix : `np.array`
        Distance matrix for the towns
    cross_platform : `bool`
        Use utils.intRounded to make sure results will be repeatable
        cross-platform. Default True

    Returns
    -------
    min_route : list
        The new route
    min_length : float
        The length of the new route

    """
    # The combinations of three places that we can split each route.
    combinations = list(itertools.combinations(range(len(route)), 3))

    min_route = route
    min_length = route_length(min_route, dist_matrix)

    for cuts in combinations:
        # The three chunks that the route is broken into based
        # on the cuts.
        c1 = route[cuts[0] : cuts[1]]
        c2 = route[cuts[1] : cuts[2]]
        c3 = route[cuts[2] :] + route[: cuts[0]]

        # Reversed chunks 2 and 3.
        rc2 = c2[::-1]
        rc3 = c3[::-1]

        # The unique permutations of all of those chunks.
        route_perms = [
            c1 + c2 + c3,
            c1 + c3 + c2,
            c1 + rc2 + c3,
            c1 + c3 + rc2,
            c1 + c2 + rc3,
            c1 + rc3 + c2,
            c1 + rc2 + rc3,
            c1 + rc3 + rc2,
        ]

        # Find the smallest of these permutations.
        for perm in route_perms:
            temp_length = route_length(perm, dist_matrix)
            if cross_platform:
                if IntRounded(temp_length) < IntRounded(min_length):
                    min_length = temp_length
                    min_route = perm
            else:
                if temp_length < min_length:
                    min_length = temp_length
                    min_route = perm

    return min_route, min_length


def tsp_convex(towns, optimize=False, niter=10):
    """Find a route through towns

    Parameters
    ----------
    towns : np.array (shape n,2)
        The points to find a path through
    optimize : bool (False)
        Optional to run the 3-optional transformation to optimize route
    niter : int (10)
        Max number of iterations to run on optimize loop.

    Returns
    -------
    indices that order towns.
    """
    hull_verts = generate_hulls(towns)
    dist_matrix = generate_dist_matrix(towns)
    route = merge_hulls(hull_verts, dist_matrix)

    if optimize:
        distance = route_length(route, dist_matrix)
        iter_count = 0
        optimized = False
        while not optimized:
            new_route, new_distance = three_opt(route, dist_matrix)
            if IntRounded(new_distance) < IntRounded(distance):
                route = new_route
                distance = new_distance
                iter_count += 1
            else:
                optimized = True
            if iter_count == niter:
                return route
    return route
