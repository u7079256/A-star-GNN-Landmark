import numpy as np
import os


def furthest_landmark(adj_matrix, num_landmark=10):
    """
    Find N landmarks, these landmarks are the furthest point pairs in the graph, the first landmark is initialized
    randomly, which may at the centric of the graph.
    :param adj_matrix: adjacent matrix of a graph
    :param num_landmark: number of landmarks chosen, set to 10 by default
    :return: A list, all element in the list indicate the index of the landmarks
    """
    # TODO
    return


def a_star_one_to_one(adj_matrix, landmark_list, from_node, to_node):
    """
    Implementation of A-star algorithm with lower_bound backbone. When calculating f(n) = g(n) + h(n). g(n) could be
    found with adjacent matrix, h(n) will be the maximum of estimated lower bound distance (The distance is calculated
    under different landmarks)
    :param adj_matrix: adjacent matrix of a graph
    :param landmark_list: Output of function "furthest landmark", which contains the candidate of landmarks
    :param from_node: the source node
    :param to_node: the target node
    :return: Distance from source node to the target node, it should be the same as Dijkstra Algorithm
    """
    # TODO
    return
