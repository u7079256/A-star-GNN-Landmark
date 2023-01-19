import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import f1_score
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv,SAGEConv
from tqdm import tqdm

from mydataset import RoadNetworkDataset


# name_data = 'Cora'
# dataset = Planetoid(root='/tmp/' + name_data, name=name_data)
# q = np.zeros(30)
# q = torch.tensor(q)
# dataset.transform = T.NormalizeFeatures()

# print(f"Number of Classes in {name_data}:", dataset.num_classes)
# print(f"Number of Node Features in {name_data}:", dataset.num_node_features)

def overall_ordering_and_checking(pred_output, gt, proportion2=0.1, proportion1=0.1,proportion0 = 0.9):
    """
    Overall ordering according to class 2 value. According to proportion given in the parameters, this function will
    classify all nodes into different class with fixed proportion.

    :param proportion2: default is 10% -> class2 10% -> class1 80% -> class 0
    :param proportion1: default is 10% -> class2 10% -> class1 80% -> class 0
    :param pred_output: Output of the model it should be converted from tensor to list first
    :param gt: ground truth of the label
    :return: it will return a list of nodes that belong to class 2, an accuracy only related to class2 and an overall F1
    """
    pred_output = np.asarray(pred_output)
    # pred_output_2 = pred_output[:,2]
    id_output_dict_2 = {}
    id_output_dict_1 = {}
    for i, node_vec in enumerate(pred_output):
        id_output_dict_1[i] = id_output_dict_2[i] = node_vec
    id_output_dict_2 = sorted(id_output_dict_2.items(), key=lambda x: x[1][0], reverse=True)
    #id_output_dict_1 = sorted(id_output_dict_1.items(), key=lambda x: (x[1][1], x[1][2], x[1][0]), reverse=True)
    pred_order = np.ones(len(pred_output))
    cut_off_value2 = int(proportion2 * len(pred_output))
    cut_off_value1 = int(proportion1 * len(pred_output))
    cut_off_value0 = int(proportion0 * len(pred_output))
    for i, ele in enumerate(id_output_dict_2):
        if i <= cut_off_value0:
            pred_order[ele[0]] = 0
    # If we meet an overwrite from 2 to 1, which means that the node is really hard for model to decide which class it
    # belongs to, then label it as 1 instead of 2, because we accept false positive, rather than false negative (more 1
    # is better than more 2)
    #for i, ele in enumerate(id_output_dict_1):
    #    if i <= cut_off_value1:
    #        pred_order[ele[0]] = 1
    gt = np.asarray(gt)
    #print(pred_order == 2)
    #print((pred_order == 2) & (gt == 2))
    acc_1 = sum((pred_order == 1) & (gt == 1)) / sum(gt==1)
    acc_0 = sum((pred_order == 0) & (gt == 0)) / sum(gt==0)
    print('for class 1 remarking',sum((pred_order == 1)),'gt has',sum(gt==1),'correctly get',sum((pred_order == 1) & (gt == 1)))
    print('for class 0 remarking', sum((pred_order == 0)), 'gt has', sum(gt == 0), 'correctly get',
          sum((pred_order == 0) & (gt == 0)))
    f1 = f1_score(pred_order, gt, average='macro')
    #print(pred_order, acc_2, f1)
    return acc_1,acc_0#, f1 #,pred_order


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.dropout = 0.5
        self.hid = 16
        self.in_head = 8
        self.out_head = 4
        self.out_dim = 191
        self.conv1 = GATConv(192, 2, heads=1, edge_dim=1)
        self.conv2 = GATConv(self.hid * self.in_head, 2,
                             heads=self.out_head, edge_dim=1, concat=False)
        self.sage1 = SAGEConv(191,191)
        self.sage2 = SAGEConv(1,2)
        self.fc1 = nn.Linear(in_features=self.out_dim, out_features=2)
        #self.fc2 = nn.Linear(in_features=32, out_features=3)
        # self.fc3 = nn.Linear(in_features=16, out_features=3)

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_attr

        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        #x = self.sage1(x,edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.sage2(x,edge_index)
        # print(type(x))
        # print(x)
        # x = F.dropout(x, p=0.5)
        # print(x.shape)
        x = self.conv1(x, edge_index)

        # print(x.shape)
        #out_1 = F.elu(out_1)
        #out_1 = torch.add(out_1,x)
        #out_1 = F.dropout(out_1, p=0.5)
        #x = out_1
        # x = F.dropout(x, p=0.5)
        #out_2 = self.conv2(out_1, edge_index, edge_features)
        # print(x.shape)
        #out_2 = F.elu(out_2)
        #out_2 = torch.add(out_2,x)
        #out_2 = F.dropout(out_2, p=0.5)
        #x = out_2
        #x = self.fc1(x)
        # x = F.dropout(x, p=0.5)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.dropout(x, p=0.5)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.dropout(x, p=0.5)
        # x = F.relu(x)
        return x  # F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

model = GAT().to(device)
# data = dataset[0].to(device)

torch.manual_seed(144)
torch.cuda.manual_seed(144)

dataset = RoadNetworkDataset(root="data/", raw_dir='t1',
                             pre_transform=T.OneHotDegree(max_degree=190, cat=False))
train_dataset = RoadNetworkDataset(root="data/", raw_dir='t1',
                                   pre_transform=T.OneHotDegree(max_degree=190, cat=False))
label_0 = 0
label_1 = 0
label_2 = 0
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

for _, b in enumerate(train_loader):
    label_0 += sum(b.y.numpy() == 0)
    label_1 += sum(b.y.numpy() == 1)
    #label_2 += sum(b.y.numpy() == 2)
weights = [label_0, label_1]#, label_2]

print(weights)
weights = torch.tensor(weights, dtype=torch.float32)
weights = weights / weights.sum()
weights = 1.0 / weights
weights = weights / weights.sum()
#weight = [0.0605, 0.4962, 0.4434]
#weights = [1, 7.96]
print(weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
weights = torch.from_numpy(np.asarray(weights)).to(torch.float64)
#weights = torch.unsqueeze(weights,1)
criterion = nn.CrossEntropyLoss(reduction='mean',weight=weights).to(device)
#torch.from_numpy(np.array(weights)).to(torch.float64)).to(
    #device)
model.train()
for epoch in tqdm(range(10000)):
    losses = 0
    model.train()
    for _, batch in enumerate(train_loader):
        batch = batch.to(device)
        model.train()
        optimizer.zero_grad()
        out = model(batch)
        out = out.to(torch.float64)
        #out = torch.squeeze(out,1)
        #out = torch.unsqueeze(out,0)
        y = batch.y.to(torch.int64)
        #y = torch.unsqueeze(y,1)
        #print(out.shape)
        #print(y.shape)
        #out = out#.Long()
        loss = criterion(out, y)
        loss.backward()
        losses += loss
        optimizer.step()
    if epoch % 50 == 0:
        print('training loss', losses)
        model.eval()
        total_correct = 0
        total_sample = 0
        total_val_loss = 0
        total_training_correct = 0
        total_training_sample = 0
        total_training_correct_1 = 0
        total_training_correct_2 = 0
        total_training_correct_0 = 0
        total_training_1 = 0
        total_training_2 = 0
        total_training_0 = 0
        total_val_correct_1 = 0
        total_val_correct_2 = 0
        total_val_correct_0 = 0
        total_val_1 = 0
        total_val_2 = 0
        total_val_0 = 0
        training_pred = []
        training_gt = []
        testing_gt = []
        testing_pred = []
        pred_node_vec_training = []
        pred_node_vec_testing = []
        gt_node_vec_training = []
        gt_node_vec_testing = []
        total_training_pred_0 = 0
        total_training_pred_1 = 0
        total_training_pred_2 = 0
        total_testing_pred_0 = 0
        total_testing_pred_1 = 0
        total_testing_pred_2 = 0
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(train_loader):
                batch = batch.to(device)
                output = model(batch)
                pred = output.argmax(dim=1)
                correct = (pred == batch.y).sum().item()
                total_training_correct_1 += ((pred == 1) & (batch.y == 1)).sum().item()
                total_training_1 += (batch.y == 1).sum().item()
                #total_training_correct_2 += ((pred == 2) & (batch.y == 2)).sum().item()
                #total_training_2 += (batch.y == 2).sum().item()
                total_training_correct_0 += ((pred == 0) & (batch.y == 0)).sum().item()
                total_training_0 += (batch.y == 0).sum().item()
                total_training_correct += correct
                total_training_sample += len(batch.y)
                total_training_pred_0 += (pred == 0).sum().item()
                total_training_pred_1 += (pred == 1).sum().item()
                #total_training_pred_2 += (pred == 2).sum().item()
                training_gt.extend(batch.y.cpu().detach().numpy().tolist())
                training_pred.extend(pred.cpu().detach().numpy().tolist())
                pred_node_vec_training.extend(output.cpu().detach().numpy().tolist())
                gt_node_vec_training.extend(batch.y.cpu().detach().numpy().tolist())
            acc = total_training_correct / total_training_sample
            acc1 = total_training_correct_1 / total_training_1
            #acc2 = total_training_correct_2 / total_training_2
            acc0 = total_training_correct_0 / total_training_0
            print("=================training process====================================================")
            print('training acc', acc, 'label 0', acc0, 'label 1', acc1)#, 'label 2', acc2)
            print('ground truth total 0', total_training_0, 'total 1', total_training_1)#, 'total 2', total_training_2)
            print('training correct prediction', total_training_correct, 'label 0', total_training_correct_0, 'label 1',
                  total_training_correct_1)#, 'label 2', total_training_correct_2)
            print('training prediction over all classes','class 0',total_training_pred_0,
                  'class 1',total_training_pred_1)#,'class 2',total_training_pred_2)
            f1 = f1_score(training_pred, training_gt, average='macro')
            print('training f1', f1)
            for _, batch in enumerate(loader):
                batch = batch.to(device)
                output = model(batch)
                out = output.double()
                y = batch.y
                #out = torch.squeeze(out, 1)
                #out = torch.unsqueeze(out, 0)
                y = batch.y.to(torch.int64)
                #y = torch.unsqueeze(y, 0)
                total_val_loss += criterion(out, y)
                pred = output.argmax(dim=1)
                correct = (pred == batch.y).sum().item()
                total_val_correct_1 += ((pred == 1) & (batch.y == 1)).sum().item()
                total_val_1 += (batch.y == 1).sum().item()
                #total_val_correct_2 += ((pred == 2) & (batch.y == 2)).sum().item()
                #total_val_2 += (batch.y == 2).sum().item()
                total_val_correct_0 += ((pred == 0) & (batch.y == 0)).sum().item()
                total_val_0 += (batch.y == 0).sum().item()
                total_correct += correct
                total_sample += len(batch.y)
                total_testing_pred_0 += (pred == 0).sum().item()
                total_testing_pred_1 += (pred == 1).sum().item()
                #total_testing_pred_2 += (pred == 2).sum().item()
                testing_gt.extend(batch.y.cpu().detach().numpy().tolist())
                testing_pred.extend(pred.cpu().detach().numpy().tolist())
                pred_node_vec_testing.extend(output.cpu().detach().numpy().tolist())
                gt_node_vec_testing.extend(batch.y.cpu().detach().numpy().tolist())
            acc = total_correct / total_sample
            acc1 = total_val_correct_1 / total_val_1
            #acc2 = total_val_correct_2 / total_val_2
            acc0 = total_val_correct_0 / total_val_0
            print("======================testing process ==================================")
            print('testing loss', total_val_loss)

            print('testing acc', acc, 'label 0', acc0, 'label 1', acc1, 'label 2')#, acc2)
            print('ground truth total 0', total_val_0, 'total 1', total_val_1)#, 'total 2', total_val_2)
            print('testing correct prediction', total_correct, 'label 0', total_val_correct_0, 'label 1',
                  total_val_correct_1)#, 'label 2', total_val_correct_2)
            print('testing prediction over all classes', 'class 0', total_testing_pred_0,
                  'class 1', total_testing_pred_1)#, 'class 2', total_testing_pred_2)

            f1 = f1_score(testing_pred, testing_gt, average='macro')
            print('testing f1', f1)
            print('ordering metrics training',
                  overall_ordering_and_checking(pred_node_vec_training,gt_node_vec_training),'testing',
                  overall_ordering_and_checking(pred_node_vec_testing,gt_node_vec_testing))
