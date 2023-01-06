import numpy as np
import dijkstra_tools


def furthest_landmark(adj_matrix, num_landmark=10):
    """
    Find N landmarks, these landmarks are the furthest point pairs in the graph, the first landmark is initialized
    randomly, which may at the centric of the graph.
    :param adj_matrix: adjacent matrix of a graph
    :param num_landmark: number of landmarks chosen, set to 10 by default
    :return: A list, all element in the list indicate the index of the landmarks
    """
    candidate_index = [i for i in range(len(adj_matrix))]
    initial_landmark = np.random.choice(candidate_index, 1)
    initial_landmark = initial_landmark[0]
    print(initial_landmark)
    landmark_list = [initial_landmark]
    sum_distance = np.zeros(len(adj_matrix))
    next_landmark = initial_landmark
    for i in range(num_landmark - 1):
        landmark_distance = dijkstra_tools.dijkstra_one_to_all(adj_matrix, from_node=next_landmark)
        landmark_distance = [-1 * np.inf if i == 0 else i for i in landmark_distance]
        sum_distance += np.asarray(landmark_distance)
        furthest_distance = max(sum_distance)
        next_landmark = np.argwhere(sum_distance == furthest_distance)[0][0]
        landmark_list.append(next_landmark)
    return landmark_list


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
    res_distance = 0
    distance_stack = [np.inf for i in range(len(adj_matrix))]
    distance_stack[from_node] = 0
    current_node = from_node
    landmark_distance_precompute = {}
    for landmark in landmark_list:
        distance_ = dijkstra_tools.dijkstra_one_to_all(adj_matrix, from_node=landmark)
        landmark_distance_precompute[landmark] = distance_

    while distance_stack[to_node] == np.inf:
        connection_status = adj_matrix[current_node]
        connected_node = [i for i in range(len(connection_status)) if connection_status[i] != 0]
        # print(connected_node)
        f_t_min = np.inf
        next_candidate = -1
        for node_index in connected_node:
            g_t = distance_stack[current_node] + adj_matrix[current_node][node_index]
            h_t = -1 * np.inf
            for landmark in landmark_list:
                distance_reload = landmark_distance_precompute[landmark]
                tri_diff = abs(distance_reload[node_index] - distance_reload[to_node])
                if tri_diff > h_t:
                    h_t = tri_diff
            f_t = g_t + h_t
            if f_t < f_t_min:
                next_candidate = node_index
                f_t_min = f_t
        distance_stack[next_candidate] = distance_stack[current_node] + adj_matrix[current_node][next_candidate]
        current_node = next_candidate
        res_distance = f_t_min

    return res_distance


def minimum_node(distance_stack, minimum_dist, close_set):
    next_node = -1
    for i in range(len(distance_stack)):
        if distance_stack[i] != np.inf and close_set[i] == 0:
            with open('di.txt', 'a') as f:
                f.writelines('searching at ' + str(i) + '\n');
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
    l = furthest_landmark(g, 4)
    print(furthest_landmark(g, 4))
    print(a_star_one_to_one(g, l, 0, 1))
    print(a_star_one_to_one(g, l, 0, 2))
    print(a_star_one_to_one(g, l, 0, 3))
    print(a_star_one_to_one(g, l, 0, 4))
    print(a_star_one_to_one(g, l, 0, 5))
    print(a_star_one_to_one(g, l, 0, 6))
    print(a_star_one_to_one(g, l, 0, 7))
    print(a_star_one_to_one(g, l, 0, 8))
