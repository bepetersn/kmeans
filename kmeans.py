#!/usr/bin/env python

import sys
import csv
import math
import random
import numpy as np
from pprint import pprint

GOOD_ENOUGH = 0.5

def main():
    
    if len(sys.argv) < 2:
        print("Usage: kmeans.py somedata.csv")
        exit(1)

    data = []
    with open(sys.argv[1]) as f:
        reader = csv.reader(f)
        for row in reader:
            if (row[0] and row[1] and row[2] and row[3]):
                data.append(np.array([
                    float(row[1]), 
                    float(row[2]), 
                    float(row[3])
                ]))

    pprint(data)
    k_means(data, 3)


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
    min = math.inf
    min_i = None
    for i, rep in enumerate(reps):
        result = get_euclidean_distance(rep, vector)
        if result < min:
            min = result
            min_i = i
    return min_i


def j_clust(vectors, groups, reps):
    value = 0
    for i, vector in enumerate(vectors):
        # note: vector op
        value += math.sqrt(get_norm(vector - reps[groups[i]]))
    return value / len(vectors)


def k_means(vectors, k, reps=None):
    # Given a list of n vectors, xi, … , xn, and k, the number of groups to form
    # Choosing initial group representatives is optional
    if reps is None:
        reps = vectors[0:k+1]
    # Choose initial groups for each vector
    groups = [random.randint(1, k) for _ in vectors]
    while True:
        for vector, i in enumerate(vectors):
            groups[i] = get_closest_representative(vector, reps)
            reps[i] = get_vectors_average(vectors)
        if j_clust(vectors, groups, reps) <= GOOD_ENOUGH:
            break
    return groups


            
      

if __name__ == "__main__":
    main()
