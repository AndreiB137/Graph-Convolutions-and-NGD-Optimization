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
