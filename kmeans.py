#!/usr/bin/env python

import sys
import csv
import math
import random
import copy
import numpy as np
from pprint import pprint
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

GOOD_ENOUGH = 0.0001
NUM_CLUSTERS = 3

def main():
    
    if len(sys.argv) != 2:
        print("Usage: ./kmeans.py somedata.csv")
        exit(1)

    data = []

    with open(sys.argv[1]) as f:
        csv_reader = csv.reader(f)
        row_iter = iter(csv_reader)
        next(row_iter) # Skip the first row
        for row in row_iter:
            if (row[0] and row[1] and row[2] and row[3]):
                data.append(_parse_vector(row))
            else:
                break

        next(row_iter) # Skip the header in this row
        initial_centers = [ # Grab the last 3 rows
            _parse_vector(next(row_iter)), 
            _parse_vector(next(row_iter)), 
            _parse_vector(next(row_iter))
        ]

    # Convert the data to a 600x3 matrix
    np_data = np.array(data)

    # Run the k-means algorithm to get the
    # group assignments
    groups, new_centers = \
        k_means(np_data, NUM_CLUSTERS, initial_centers)

    # Segregate the vectors into groups by the group 
    # assignments we generated above
    grouped_vectors = []
    for group_id in range(0, NUM_CLUSTERS):
        grouped = [v for (i, v) in enumerate(np_data) \
                     if groups[i] == group_id]
        grouped_vectors.append(grouped)
            
    # Start plotting
    fig = pyplot.figure()
    ax = Axes3D(fig)
    colors = ['yellow', 'cyan', 'red']

    # First plot the new centers
    np_new_centers = np.array(new_centers)
    np_new_centers_t = np_new_centers.transpose()
    ax.scatter(
        np_new_centers_t[0],
        np_new_centers_t[1],
        np_new_centers_t[2], color="green", marker="X",
                             zorder=5
    )

    # Then the old centers
    np_initial_centers = np.array(initial_centers)
    np_initial_centers_t = np_initial_centers.transpose()
    ax.scatter(
        np_initial_centers_t[0],
        np_initial_centers_t[1],
        np_initial_centers_t[2], color="blue", marker="X",
                                 zorder=5
    )

    # Plot each cluster one at a time
    for i, vector_group in enumerate(grouped_vectors):
        np_vector_group = np.array(vector_group)
         # below: matplotlib expects axes separated out
        np_vector_group_t = np_vector_group.transpose()
        ax.scatter(
            np_vector_group_t[0], 
            np_vector_group_t[1], 
            np_vector_group_t[2], color=colors[i])
    
    # Finish plotting
    matplotlib.use('TkAgg')
    pyplot.show()


def _parse_vector(row): 
    return [
        float(row[1]), float(row[2]), float(row[3])
    ]


def get_norm(vector):
    return math.sqrt(vector.dot(vector))


def get_euclidean_distance(vector1, vector2):
    return get_norm(vector1 - vector2)  # note: vector op


def get_vectors_average(vectors):
    new_vector = [0] * len(vectors[0])
    for vector in vectors:
        for i, element in enumerate(vector):
            new_vector[i] += element
    for i in range(len(vectors[0])):
        new_vector[i] = new_vector[i] / len(vectors)
    return new_vector


def get_closest_representative(vector, reps):
    """Find the closest representative to a vector 
       from among representatives and return its 
       index in reps. 

    Args:
        vector (numpy.array): A vector to compare
        against the representative vectors in reps
        reps (list[numpy.array]): A list of vectors,
        to be compared against vector

    Returns:
        int: Return the index of the closest 
        representative vector within reps
    """
    min = math.inf
    min_i = None
    for i, rep in enumerate(reps):
        result = get_euclidean_distance(rep, vector)
        if result < min:
            min = result
            min_i = i
    return min_i


def get_j_clust(vectors, groups, reps):
    value = 0
    for i, vector in enumerate(vectors):
        # note: vector op
        value += math.sqrt(get_norm(vector - reps[groups[i]]))
    return value / len(vectors)


def k_means(vectors, k, reps):
    # Given a list of n vectors, xi, … , xn, and k, the number of 
    # groups to form, and reps, the initial choice of vector 
    # representatives fo each group / cluster

    # Choose initial groups for each vector
    groups = list(range(0, k))
    group_assignments = [random.randint(0, k-1) for _ in vectors]
    # NOTE: Do the below so we don't muddle the original reps
    group_reps = copy.deepcopy(reps)  
    old_j_clust = j_clust = math.inf
    while True:
        print("group assignments: {}".format(group_assignments))
        print("group_reps: {}".format(group_reps))
        print("j_clust: {}".format(j_clust))
        for i, vector in enumerate(vectors):
            group_assignments[i] = \
                get_closest_representative(vector, group_reps)
        for group in groups:
            vector_indices_in_grp = \
                [i for (i, g) in enumerate(group_assignments) \
                        if g == group]
            group_reps[group] = get_vectors_average(
                [v for (i, v) in enumerate(vectors) \
                        if i in vector_indices_in_grp]
            )
        old_j_clust = j_clust
        j_clust = get_j_clust(vectors, group_assignments, group_reps)
        if (old_j_clust - j_clust) <= GOOD_ENOUGH:
            print("diff: {}".format(old_j_clust - j_clust))
            break
    return group_assignments, group_reps
      

if __name__ == "__main__":
    main()
