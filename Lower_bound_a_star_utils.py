import os
import pickle

import utils

STORING_PATH = 'distance_data'
READ_PATH = 'graph'


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def pickle_training_data(root_dir=READ_PATH, target_path=STORING_PATH):
    """
    It saves training data which will be used later on the disk
    """
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            #csv_path = os.path.join(READ_PATH, file)
            #print(csv_path)
            adj, spa = utils.read_graph(file)
            feature_dict = utils.feature_matrix_extraction(adj_matrix=adj,
                                                           boundary_nodes=utils.boundary_node_detection(adj))
            save_obj(feature_dict, os.path.join(target_path, file.split('.')[0]))
            # check_for_maintenance
            reload_dict = load_obj(os.path.join(target_path, file.split('.')[0]))
            assert reload_dict == feature_dict


if __name__ == '__main__':
    pickle_training_data()
