import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from tqdm import tqdm

from mydataset import RoadNetworkDataset

# name_data = 'Cora'
# dataset = Planetoid(root='/tmp/' + name_data, name=name_data)
# q = np.zeros(30)
# q = torch.tensor(q)
# dataset.transform = T.NormalizeFeatures()

#print(f"Number of Classes in {name_data}:", dataset.num_classes)
#print(f"Number of Node Features in {name_data}:", dataset.num_node_features)


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(1, self.hid, heads=self.in_head, dropout=0.6,edge_dim=1)
        self.conv2 = GATConv(self.hid * self.in_head, 3, concat=False,
                             heads=self.out_head, dropout=0.6,edge_dim=1)

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index , data.edge_attr
        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        # print(type(x))
        # print(x)
        x = F.dropout(x, p=0.5)
        x = self.conv1(x, edge_index,edge_features)
        x = F.elu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv2(x, edge_index,edge_features)

        return x  # F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

model = GAT().to(device)
# data = dataset[0].to(device)
dataset = RoadNetworkDataset(root="data/", raw_dir='testing')
train_dataset = RoadNetworkDataset(root="data/", raw_dir='training')
label_0 = 0
label_1 = 0
label_2 = 0
loader = DataLoader(dataset, batch_size=16, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for _, b in enumerate(train_loader):
    label_0 += sum(b.y.numpy() == 0)
    label_1 += sum(b.y.numpy() == 1)
    label_2 += sum(b.y.numpy() == 2)
weights = [label_0,label_1,label_2]
print(weights)
weights = torch.tensor(weights, dtype=torch.float32)
weights = weights / weights.sum()
weights = 1.0 / weights
weights = weights / weights.sum()
print(weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(reduction='mean',weight=torch.from_numpy(np.array(weights)).to(torch.float64)).to(device)
model.train()
for epoch in tqdm(range(100)):
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
        print(losses)
        model.eval()
        total_correct = 0
        total_sample = 0
        with torch.no_grad():
            for _, batch in enumerate(loader):
                batch = batch.to(device)
                output = model(batch)
                pred = output.argmax(dim=1)
                correct = (pred == batch.y).sum().item()
                total_correct += correct
                total_sample += len(batch.y)
            acc = total_correct / total_sample
            print('Accuracy: {:.4f}'.format(acc))
            print('correct', total_correct)

