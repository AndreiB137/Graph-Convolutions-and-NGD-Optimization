class GCN(torch.nn.Module):
    def __init__(self, eps, num_nodes, dataset):
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
