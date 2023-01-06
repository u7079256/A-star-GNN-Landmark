import numpy as np
import os


def dijkstra_one_to_one(adj_matrix, from_node, to_node):
    """
    Basic Dijkstra algo implementation
    :param adj_matrix: adj matrix which contains information related to the graph
    :param from_node: the source node index
    :param to_node: the target node index
    :return: Distance between source and target, which should be the shortest path
    """
    close_set = np.zeros(len(adj_matrix))
    distance_stack = [np.inf for i in range(len(adj_matrix))]
    distance_stack[from_node] = 0
    while close_set[to_node] == 0:
        next_node = minimum_node(distance_stack, np.inf, close_set)
        connection_status = adj_matrix[next_node]
        for i in range(len(connection_status)):
            if connection_status[i] != 0 and distance_stack[i] > distance_stack[next_node] + connection_status[i]:
                distance_stack[i] = distance_stack[next_node] + connection_status[i]
    print(close_set)
    return distance_stack[to_node]


def dijkstra_one_to_all(adj_matrix, from_node):
    """
    Basic Dijkstra algo implementation
    :param adj_matrix: adj matrix which contains information related to the graph
    :param from_node: the source node index
    :return: Distance between source and target, which should be the shortest path
    """
    close_set = np.zeros(len(adj_matrix))
    distance_stack = [np.inf for i in range(len(adj_matrix))]
    distance_stack[from_node] = 0
    while 0 in close_set:
        next_node = minimum_node(distance_stack, np.inf, close_set)
        connection_status = adj_matrix[next_node]
        for i in range(len(connection_status)):
            if connection_status[i] != 0 and distance_stack[i] > distance_stack[next_node] + connection_status[i]:
                distance_stack[i] = distance_stack[next_node] + connection_status[i]

    return distance_stack


def minimum_node(distance_stack, minimum_dist, close_set):
    next_node = -1
    for i in range(len(distance_stack)):
        if distance_stack[i] != np.inf and close_set[i] == 0:
            with open('di.txt', 'a') as f:
                f.writelines('searching at ' + str(i)+'\n')
            if distance_stack[i] < minimum_dist:
                minimum_dist = distance_stack[i]
                next_node = i
    close_set[next_node] = 1
    return next_node


if __name__ == '__main__':
    g = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
         [4, 0, 8, 0, 0, 0, 0, 11, 0],
         [0, 8, 0, 7, 0, 4, 0, 0, 2],
         [0, 0, 7, 0, 9, 14, 0, 0, 0],
         [0, 0, 0, 9, 0, 10, 0, 0, 0],
         [0, 0, 4, 14, 10, 0, 2, 0, 0],
         [0, 0, 0, 0, 0, 2, 0, 1, 6],
         [8, 11, 0, 0, 0, 0, 1, 0, 7],
         [0, 0, 2, 0, 0, 0, 6, 7, 0]
         ]
    print(dijkstra_one_to_one(g, 0, 1))
    print(dijkstra_one_to_one(g, 0, 2))
    print(dijkstra_one_to_one(g, 0, 3))
    print(dijkstra_one_to_one(g, 0, 4))
    print(dijkstra_one_to_one(g, 0, 5))
    print(dijkstra_one_to_one(g, 0, 6))
    print(dijkstra_one_to_one(g, 0, 7))
    print(dijkstra_one_to_one(g, 0, 8))
    #print(dijkstra_one_to_all(g, 3))
    #print(dijkstra_one_to_all(g, 0))
    #print(dijkstra_one_to_all(g, 4))
    #print(dijkstra_one_to_all(g, 4))
