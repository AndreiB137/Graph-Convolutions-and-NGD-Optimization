lossf = nn.NLLLoss()
loss_table = []
train_data = split_graph_data(data)
model = GCN(torch.tensor([2e-2]), len(torch.nonzero(data.train_mask)), data)
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