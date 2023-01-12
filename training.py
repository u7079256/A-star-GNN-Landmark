import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import f1_score
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from tqdm import tqdm

from mydataset import RoadNetworkDataset


# name_data = 'Cora'
# dataset = Planetoid(root='/tmp/' + name_data, name=name_data)
# q = np.zeros(30)
# q = torch.tensor(q)
# dataset.transform = T.NormalizeFeatures()

# print(f"Number of Classes in {name_data}:", dataset.num_classes)
# print(f"Number of Node Features in {name_data}:", dataset.num_node_features)


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 16
        self.in_head = 16
        self.out_head = 8
        self.out_dim = 3
        self.conv1 = GATConv(191, self.hid, heads=self.in_head, dropout=0.5, edge_dim=1)
        self.conv2 = GATConv(self.hid*self.in_head , self.out_dim,
                             heads=self.out_head, dropout=0.5, edge_dim=1,concat=False)
        self.fc1 = nn.Linear(in_features=self.out_dim, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=3)
        #self.fc3 = nn.Linear(in_features=16, out_features=3)

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_attr

        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        # print(type(x))
        # print(x)
        # x = F.dropout(x, p=0.5)
        # print(x.shape)
        x = self.conv1(x, edge_index, edge_features)
        #print(x.shape)
        x = F.elu(x)
        # x = F.dropout(x, p=0.5)
        x = self.conv2(x, edge_index, edge_features)
        #print(x.shape)
        x = F.elu(x)
        #x = self.fc1(x)
        #x = F.dropout(x, p=0.5)
        #x = F.relu(x)
        #x = self.fc2(x)
        #x = F.dropout(x, p=0.5)
        #x = F.relu(x)
        #x = self.fc3(x)
        #x = F.dropout(x, p=0.5)
        #x = F.relu(x)
        return x  # F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

model = GAT().to(device)
# data = dataset[0].to(device)

torch.manual_seed(144)
torch.cuda.manual_seed(144)

dataset = RoadNetworkDataset(root="data/", raw_dir='testing',
                             pre_transform=T.OneHotDegree(max_degree=190, cat=False))
train_dataset = RoadNetworkDataset(root="data/", raw_dir='training',
                                   pre_transform=T.OneHotDegree(max_degree=190, cat=False))
label_0 = 0
label_1 = 0
label_2 = 0
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

for _, b in enumerate(train_loader):
    label_0 += sum(b.y.numpy() == 0)
    label_1 += sum(b.y.numpy() == 1)
    label_2 += sum(b.y.numpy() == 2)
weights = [label_0, label_1, label_2]
print(weights)
weights = torch.tensor(weights, dtype=torch.float32)
weights = weights / weights.sum()
weights = 1.0 / weights
weights = weights / weights.sum()
print(weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(reduction='mean', weight=torch.from_numpy(np.array(weights)).to(torch.float64)).to(
    device)
model.train()
for epoch in tqdm(range(1000)):
    losses = 0
    model.train()
    for _, batch in enumerate(train_loader):
        batch = batch.to(device)
        model.train()
        optimizer.zero_grad()
        out = model(batch)
        y = batch.y.to(torch.int64)
        out = out.double()
        loss = criterion(out, y)
        loss.backward()
        losses += loss
        optimizer.step()
    if epoch % 5 == 0:
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
        with torch.no_grad():
            for _, batch in enumerate(train_loader):
                batch = batch.to(device)
                output = model(batch)
                pred = output.argmax(dim=1)
                correct = (pred == batch.y).sum().item()
                total_training_correct_1 += ((pred == 1) & (batch.y == 1)).sum().item()
                total_training_1 += (batch.y == 1).sum().item()
                total_training_correct_2 += ((pred == 2) & (batch.y == 2)).sum().item()
                total_training_2 += (batch.y == 2).sum().item()
                total_training_correct_0 += ((pred == 0) & (batch.y == 0)).sum().item()
                total_training_0 += (batch.y == 0).sum().item()
                total_training_correct += correct
                total_training_sample += len(batch.y)
                training_gt.extend(batch.y.cpu().detach().numpy().tolist())
                training_pred.extend(pred.cpu().detach().numpy().tolist())
            acc = total_training_correct / total_training_sample
            acc1 = total_training_correct_1 / total_training_1
            acc2 = total_training_correct_2 / total_training_2
            acc0 = total_training_correct_0 / total_training_0
            print('training acc', acc, 'label 0', acc0, 'label 1', acc1, 'label 2', acc2)
            print('total 0', total_training_0, 'total 1', total_training_1, 'total 2', total_training_2)
            print('training correct', total_training_correct, 'label 0', total_training_correct_0, 'label 1',
                  total_training_correct_1, 'label 2', total_training_correct_2)
            f1 = f1_score(training_pred, training_gt, average='macro')
            print('training f1', f1)
            for _, batch in enumerate(loader):
                batch = batch.to(device)
                output = model(batch)
                out = output.double()
                y = batch.y.to(torch.int64)
                total_val_loss += criterion(out, y)
                pred = output.argmax(dim=1)
                correct = (pred == batch.y).sum().item()
                total_val_correct_1 += ((pred == 1) & (batch.y == 1)).sum().item()
                total_val_1 += (batch.y == 1).sum().item()
                total_val_correct_2 += ((pred == 2) & (batch.y == 2)).sum().item()
                total_val_2 += (batch.y == 2).sum().item()
                total_val_correct_0 += ((pred == 0) & (batch.y == 0)).sum().item()
                total_val_0 += (batch.y == 0).sum().item()
                total_correct += correct
                total_sample += len(batch.y)
                testing_gt.extend(batch.y.cpu().detach().numpy().tolist())
                testing_pred.extend(pred.cpu().detach().numpy().tolist())
            acc = total_correct / total_sample
            acc1 = total_val_correct_1 / total_val_1
            acc2 = total_val_correct_2 / total_val_2
            acc0 = total_val_correct_0 / total_val_0
            print('training acc', acc, 'label 0', acc0, 'label 1', acc1, 'label 2', acc2)
            print('total 0', total_val_0, 'total 1', total_val_1, 'total 2', total_val_2)
            print('training correct', total_correct, 'label 0', total_val_correct_0, 'label 1',
                  total_val_correct_1, 'label 2', total_val_correct_2)
            print('val loss', total_val_loss)
            f1 = f1_score(testing_pred, testing_gt, average='macro')
            print('testing f1', f1)
