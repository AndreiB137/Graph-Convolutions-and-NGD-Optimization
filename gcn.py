import networkx as nx
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import sys
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import ClusterLoader
from torch_geometric.loader import ClusterData
from torch_geometric.utils import to_networkx
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), node_size = 1, with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

torch.manual_seed(300)

def split_graph_data(G, num_clusters = 50):
    cluster_data = ClusterData(
        G, num_parts=num_clusters, recursive=False, save_dir = "Split_graph_cora"
    ) 

    return cluster_data

def sample(cluster_data, batch_size = 10):
    dynamic_loader = ClusterLoader(
        cluster_data,
        batch_size = batch_size,
        shuffle = True
    )
    return iter(dynamic_loader)


dataset = Planetoid(root = "cora_simple", name = "Cora")
data = dataset[0]
#print(f'Dataset: {dataset}:')
#print(f'Number of graphs: {len(dataset)}')
#print(f'Number of nodes: {len(data.x)}')
#print(f'Number of edges: {data.edge_index.shape[1]}')
#print(f'Number of features: {dataset.num_features}')
#print(f'Number of classes: {dataset.num_classes}')

# G = to_networkx(data, to_undirected=True)
# visualize_graph(G, color = data.y)

mask = torch.zeros((len(data.y),), dtype = torch.bool)
mask[torch.randint(0, len(data.y), (50,))] = True

class GCN(torch.nn.Module):
    def __init__(self, eps, num_nodes):
        super().__init__()
        torch.manual_seed(1234)
        self.num_train_nodes = num_nodes
        self.eps = eps
        self.conv1 = GCNConv(dataset.num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim = 1)
        self.dropout = nn.Dropout(p = 0.5)
        self.proj = nn.Linear(64, dataset.num_classes)
        self.x1_tilda = None
        self.x1 = None
        self.x2_tilda = None
        self.x2 = None

    def forward(self, x, edge_index):
        x1_tilda, x1 = self.conv1(x, edge_index)
        x1_tilda.detach()
        x1 = self.relu(x1)
        self.x1_tilda = x1_tilda
        x1 = self.dropout(x1)
        x1.retain_grad()
        self.x1 = x1
        x2_tilda, x2 = self.conv2(x1, edge_index)
        x2_tilda.detach()
        # print(x2_tilda.shape)
        self.x2_tilda = x2_tilda
        x2 = self.relu(x2)
        x2.retain_grad()
        self.x2 = x2

        output = self.log_softmax(self.proj(x2))

        return output

def inverse(X, eps):
    return torch.linalg.inv(X + torch.pow(eps, -0.5) * torch.eye(X.shape[0]))

def update_weights(model):
    with torch.no_grad():
        u_k1 = 1 / model.num_train_nodes * (model.x1.grad * model.x1).t() @ (model.x1.grad * model.x1)
        v_k1 = 1 / model.num_train_nodes * (model.x1_tilda.t() @ model.x1_tilda)
        u_k2 = 1 / model.num_train_nodes * (model.x2.grad * model.x2).t() @ (model.x2.grad * model.x2)
        v_k2 = 1 / model.num_train_nodes * (model.x2_tilda.t() @ model.x2_tilda)
        model.conv1.lin.weight.grad = (inverse(u_k1, model.eps) @ model.conv1.lin.weight.grad @ inverse(v_k1, model.eps))
        model.conv2.lin.weight.grad = (inverse(u_k2, model.eps) @ model.conv2.lin.weight.grad @ inverse(v_k2, model.eps))

lossf = nn.NLLLoss()
loss_table = []
train_data = split_graph_data(data)
model = GCN(torch.tensor([2e-2]), len(torch.nonzero(data.train_mask)))
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, weight_decay=5e-4, momentum=0.9)


def train(epochs, model, optimizer):
    for i in tqdm(range(epochs)):
        # print(torch.max(batch.edge_index))
        y = model(data.x, data.edge_index)
        indices_train = torch.nonzero(data.train_mask)
        model.num_train_nodes = len(indices_train)
        # print(len(indices_train) / len(batch.y) * 100)
        loss = lossf(y[indices_train].squeeze(dim = 1), data.y[indices_train].squeeze())
        loss_table.append(loss.detach().cpu().item())
        if i % 5 == 0:
            print(f"The loss at epoch {i} is {loss.detach().cpu().item():.5f}")
            indices_val = data.val_mask
            with torch.no_grad():
                acc = len(torch.nonzero(torch.argmax(y[indices_val].squeeze(dim = 1), dim = 1) == data.y[indices_val])) / len(indices_val) * 100
                print(f"The accuracy is {acc:.2f}")
        loss.backward()
        #update_weights(model)
        optimizer.step()
        optimizer.zero_grad()

train(200, model, optimizer)

def accuracy_test(model):
    cnt = 0
    model.eval()
    y = model(data.x, data.edge_index)
    with torch.no_grad():
        for i in range(len(y)):
            if torch.argmax(y[i]) == data.y[i]:
                cnt += 1
    
    return cnt / len(y) * 100

acr = accuracy_test(model)


print(f"The final accuracy of the model is {acr:.2f}")

sys.exit()

def visualise_test(y_axis, x_value):
    x_axis = [i for i in range(x_value)]
    plt.plot(x_axis, y_axis)
    plt.show()

visualise_test(loss_table, 200)

sys.exit()

for i in tqdm(range(29, 30)):
    model = GCN()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-2, weight_decay=5e-4)
    train(i, model, optimizer)
    acc = accuracy_test(model)
    accuracy_table.append(accuracy_test(model))

print(accuracy_table)