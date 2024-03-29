import os
from typing import Union, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

import Lower_bound_a_star_utils
from graph_generate import read_graph
from utils import feature_matrix_extraction, boundary_node_detection
import torch_geometric.transforms as T


class RoadNetworkDataset(Dataset):
    def __init__(self, root, raw_dir='graph', test=False, transform=None, pre_transform=None, proportion=0.1):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.proportion = proportion
        self.store_dir = root
        self.graph_dir = raw_dir
        graph_dir = self.graph_dir
        root, dirs, files = next(os.walk(graph_dir, topdown=True))
        self.files = files
        #print('kkk', (files))
        self.length = len(files)
        self.mean_degree = 0
        super(RoadNetworkDataset, self).__init__(root, transform, pre_transform)
        # shutil.rmtree(os.path.join(graph_dir, 'processed'))
        # shutil.rmtree(os.path.join(graph_dir, 'raw'))

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """ If this file exists in raw_dir, the download is not triggered.
                    (The download func. is not implemented here)
            These files' content is separated with ',' which is similar to csv files
        """
        return self.files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        # return 'road.pt'
        # print('processed', [f'data_{i.split(".")[0]}.pt' for i in list(self.files)])
        #print('self', self.files)
        # print(len([f'data_{i}.pt' for i in range(1271)]))
        #print('lllll', len(self))

        return [f'data_{i}.pt' for i in range(self.length)]

    def download(self):
        """
        self made data, no url to download
        """
        # raise Exception('Please check the raw data folder and make sure there is no missing files')
        pass

    def process(self):
        """
        Load corresponding files, and convert information into Data format, which could be used in network training

        """

        graph_dir = self.graph_dir
        data_list = []
        root, dirs, files = next(os.walk(graph_dir, topdown=True))
        for index, file in tqdm(enumerate(files), total=len(files)):
            # print('wyr',file,files)
            adj, spa = read_graph(file)
            mean_degree = 0
            degree_list = []
            # initial definition, our graphs do not have node feature, so initialize all node features with '1',
            # which could be regarded as a placeholder
            x = []
            edge_index_list = [i for i in range(len(adj))]
            for row in adj:
                degree = (row != 0).sum()
                x.append(degree)
                degree_list.append(degree)
            #print('max degree',max_degree)
            mean_degree = np.mean(degree_list)
            self.mean_degree = mean_degree
            x = np.asarray(x)
            #x = x/max_degree
            x = torch.from_numpy(x)

            from_list = []
            to_list = []
            edge_attr = []

            for ele in spa:
                # the spare matrix form only contain a -> b (where id_a <= id_b), which requires to repeat twice
                # because edge_index regard edges as directed.
                from_list.extend([int(ele[0]), int(ele[1])])
                to_list.extend([int(ele[1]), int(ele[0])])
                edge_attr.extend([[float(ele[2])], [float(ele[2])]])
            edge_index = np.asarray([from_list, to_list])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            # Note that normally the GAT layer does not take edge weight into convolution because it calculates
            # a dynamic distance between two nodes. We put the distance into node attribute as a naive conversion
            # on this problem.
            edge_attr = np.asarray(edge_attr)
            edge_attr = torch.tensor(edge_attr)
            #print(edge_index_list)
            #print(self.mean_degree)
            feature_dict = feature_matrix_extraction(adj_matrix=adj,
                                                     boundary_nodes=edge_index_list)
            #print(feature_dict)
            landmark_score = Lower_bound_a_star_utils.scoring_with_ordering(feature_dict)

            landmark_score = sorted(landmark_score.items(), key=lambda x: x[1], reverse=True)
            #print(landmark_score)
            y = np.zeros(len(adj))

            cut_off_index = int(self.proportion * len(landmark_score))
            #print(cut_off_index)
            # print('len', len(landmark_score),'cut',cut_off_index)
            for ele in landmark_score:
                # label = 2 indicates good landmark, 1 ~ bad landmarks, 0 ~ normal node
                if cut_off_index >= 0:
                    y[ele[0]] = 1
                    cut_off_index -= 1
            #print(y)
            y = torch.tensor(y)
            graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=y)
            graph.x = graph.x.to(torch.float)
            graph.x = (graph.x - graph.x.mean()) / graph.x.std()
            #print(graph.x)
            if self.pre_filter is not None:
                graph = self.pre_filter(graph)

            if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
            # print('the file', file)

            torch.save(graph,
                       os.path.join(self.processed_dir,
                                    f'data_{index+(0)}.pt'))

    def len(self) -> int:
        return self.length

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
        - Is not needed for PyG's InMemoryDataset
        """
        # print(os.path.join(self.processed_dir,
        #                   f'data_{idx}.pt'))
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data



if __name__ == '__main__':
    dataset = RoadNetworkDataset(root="data/", raw_dir='t1',pre_transform=T.OneHotDegree(max_degree=190,cat=True))
    dataset = RoadNetworkDataset(root="data/", raw_dir='testing', pre_transform=T.OneHotDegree(max_degree=190, cat=True))
    #dataset = RoadNetworkDataset(root="data/", raw_dir='training', pre_transform=T.OneHotDegree(max_degree=190, cat=True))
    sample = torch.load(os.path.join('t1//processed',
                                     f'data_0.pt'))
    #print(sample.x[0][27])
    #print(len(sample.x[0]))
    sample2 = torch.load(os.path.join('t1//processed',
                                     f'data_2.pt'))
    for ele in sample.x:
        ele = ele.numpy()
        for ele2 in ele:
            if ele2 != 0:
                print(ele2)
    print(sample.x == sample2.x)
    b = sample.x == sample2.x
    b = b.numpy()
    print((b==1).sum())
    print((sample.x == 1).sum())
    c = sample2.y
    d = torch.tensor([])
    f = torch.cat((c,d))
