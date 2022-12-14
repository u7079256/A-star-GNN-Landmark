import random
import csv
# random.seed(seed)
import numpy as np


def initialize(node_num, max_degree):
    degree_dict = {}  # id , (current_degree, max_degree)
    for id in node_num:
        id_degree = random.randint(1, max_degree)
        degree_dict[id] = (0, id_degree)
    return degree_dict


def generate_graph(nodes_num, max_weight):
    matrix = np.zeros((nodes_num, nodes_num))
    max_degree = int(0.8 * nodes_num)
    nodes = np.asarray([id for id in range(nodes_num)])
    current_degree = np.zeros(nodes_num)

    for id in range(nodes_num):
        local_degree_max = random.randint(1,max_degree)
        legal_connect = nodes[id+1:]
        sublist_degree = (current_degree < max_degree)[id+1:]
        legal_connect = legal_connect[sublist_degree]
        legal_degree = local_degree_max - current_degree[id]
        #print(legal_degree)
        if 0 < legal_degree:
            true_connect = np.random.choice(legal_connect, int(legal_degree),
                                            replace=False) if legal_degree <= len(legal_connect) else legal_connect
            current_degree[id] = local_degree_max
            for to_ele in true_connect:
                matrix[id][to_ele] = format(np.random.uniform(1, max_weight),'.4f')
                matrix[to_ele][id] = matrix[id][to_ele]
                #print(current_degree)
                current_degree[to_ele] = current_degree[to_ele] + 1

    def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)
    assert check_symmetric(matrix)
    assert np.max(current_degree) <= max_degree, 'max_degree' + str(max_degree) + "  "+str(np.max(current_degree))
    return matrix,current_degree
def store_graph(adj_matrix,file_name):
    header = ['row','col','weight']
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in range(len(adj_matrix)):
            for col in range(len(adj_matrix[0])):
                if row <= col and adj_matrix[row][col] > 0:
                    writer.writerow([row,col,adj_matrix[row][col]])
def read_graph(file_name):
    num = file_name.split('_')[0]

    adj_matrix = np.zeros((int(num),int(num)))
    f = open(file_name)
    csv_reader = csv.reader(f)
    next(csv_reader)
    sparse_matrix = []
    for line in csv_reader:
        #print(line)
        adj_matrix[int(line[0])][int(line[1])] = float(line[2])
        adj_matrix[int(line[1])][int(line[0])] = adj_matrix[int(line[0])][int(line[1])]
        sparse_matrix.append(line)


    return adj_matrix,sparse_matrix

if __name__ == '__main__':
    seed = 1224
    # max_degree = 12
    # max_weight = 60
    # nodes_num = 60
    step = 0
    # for i in range(0,10000,5):
    #     for num in range(10,1000,5):
    #         for weight in range(6,600,4):
    #             matrix, current = generate_graph(num,weight)
    #             step += 1
    #             if step % 100 == 0:
    #                 print(matrix)
    #                 print(matrix.shape)
    #                 print(current)
    # num = 6
    # weight = 4
    # matrix, current = generate_graph(num, weight)
    # print(matrix)
    # print(current)
    # file_name = "" + str(num) +'_' + str(weight) +'.csv'
    # store_graph(matrix,file_name=file_name)
    # read_graph(file_name)
    for index in range(0,20):
        for num in range(10,200,10):
            for weight in range(6,int(num*0.8),8):
                matrix,current = generate_graph(num,weight)
                file_ = str(num) +'_' + str(weight) +'_' + str(index) +'.csv'
                store_graph(matrix, file_name=file_)

