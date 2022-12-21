import numpy as np
from graph_generate import read_graph


def boundary_node_detection(adj_matrix, proportion=0.1):
    """
    Boundary of graph should have relatively low degree than nodes at the center of the
    graph. Therefore, nodes which number of degree stand at the lower 10% will be regarded
    as the boundary.

    :param proportion: the proportion that will be considered as boundary nodes
    :param adj_matrix: adjacent matrix of the graph
    :return: a list of nodes (index) that located at the edge of the graph
    """
    degree_list = []
    length = len(adj_matrix[0])
    node_num = len(adj_matrix)
    index = 0
    for row in adj_matrix:
        degree_list.append((index, length - list(row).count(0)))
        index += 1
    degree_list.sort(key=lambda x: x[1])
    top_prop = 2 if int(node_num * proportion) <= 2 else int(node_num * proportion)
    boundary_degree = degree_list[top_prop - 1][1]
    node_index = [x[0] for x in degree_list[:top_prop]]
    for i in range(top_prop, node_num):
        if degree_list[i][1] == boundary_degree:
            node_index.append(degree_list[i][0])
        else:
            break
    return node_index


def centric_node_detection(adj_matrix, proportion=0.1):
    """
    This function aims at finding nodes that lying at the center of the graph, which should have higher degree that
    other nodes.
    :param proportion: the proportion that will be considered as centric nodes
    :param adj_matrix: adjacent matrix of the graph
    :return: a list of nodes (index) that located at the edge of the graph
    """
    degree_list = []
    length = len(adj_matrix[0])
    node_num = len(adj_matrix)
    index = 0
    for row in adj_matrix:
        degree_list.append((index, length - list(row).count(0)))
        index += 1
    degree_list.sort(key=lambda x: x[1],reverse=True)
    top_prop = 2 if int(node_num * proportion) <= 2 else int(node_num * proportion)
    boundary_degree = degree_list[top_prop - 1][1]
    node_index = [x[0] for x in degree_list[:top_prop]]
    for i in range(top_prop, node_num):
        if degree_list[i][1] == boundary_degree:
            node_index.append(degree_list[i][0])
        else:
            break
    return node_index


def feature_matrix_extraction(adj_matrix,boundary_nodes,gt = False):
    """
    A FEATURE matrix which contains possible features for future use, all elements are corresponding to the estimated
    length of the node pairs based on one of the landmarks.
    :param gt: Set to true to available calculating ground truth of the distance
    :param adj_matrix:
    :param boundary_nodes:
    :return: it should be a matrix with (len(boundary_nodes),num(node_pairs)). node_pairs should be a combination of all
    remaining nodes
    """
    nodes_list = list(np.arange(len(adj_matrix)))
    potential_pairs = list(set(nodes_list) - set(boundary_nodes))
    feature_dict = {}
    for ele_from in potential_pairs:
        for ele_to in potential_pairs:
            if ele_to != ele_from:
                feature_list_dict = {}
                for landmark in boundary_nodes:
                    feature_list_dict[landmark] = abs(adj_matrix[ele_from][landmark]-adj_matrix[ele_to][landmark])
                feature_dict[(ele_from,ele_to)] = feature_list_dict

    return feature_dict


if __name__ == '__main__':
    adj, spa = read_graph('10_6_0.csv')
    print(adj)
    count = 0
    for ele in spa:
        print('index', count, ':', ele)
        count += 1
    print(boundary_node_detection(adj))
    print(centric_node_detection(adj))
    print(feature_matrix_extraction(adj_matrix=adj,boundary_nodes=boundary_node_detection(adj)))
