import torch
try:
    from torch_geometric.data import Data
    print("Successfully imported torch_geometric")
except ImportError:
    print("torch_geometric not found, treating as standard PyTorch file")

file_path = 'processed_graph.pt'

try:
    data = torch.load(file_path)
    print(f"\nLoaded data type: {type(data)}")
    print("-" * 30)

    if hasattr(data, 'keys'): # Check if it acts like a dictionary or Data object
        print("Keys/Attributes found:")
        # keys might be a method or a property depending on version
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            # in PyG Data object, keys is a property, or use keys() method
            val = data[key]
            if torch.is_tensor(val):
                print(f"  {key}: Tensor of shape {val.shape}, dtype {val.dtype}")
            else:
                print(f"  {key}: {type(val)}")
                
        # Specific check for PyG Data object structure
        if hasattr(data, 'num_nodes'):
             print(f"\nGraph Info:")
             print(f"  Number of nodes: {data.num_nodes}")
             print(f"  Number of edges: {data.num_edges}")
             print(f"  Has isolated nodes: {data.has_isolated_nodes()}")
             print(f"  Has self loops: {data.has_self_loops()}")
             print(f"  Is undirected: {data.is_undirected()}")

    else:
        print("Data content:", data)

except Exception as e:
    print(f"Error loading file: {e}")

"data/"