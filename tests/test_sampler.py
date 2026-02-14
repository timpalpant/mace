import torch
import pytest
from unittest.mock import MagicMock, patch
from mace.data.sampler import DynamicBatcher
from mace.tools.torch_geometric.data import Data

class MockDataset(torch.utils.data.Dataset):
    def __init__(self, node_counts, edge_counts=None):
        self.node_counts = node_counts
        self.edge_counts = edge_counts

    def __getitem__(self, idx):
        data = Data()
        data.x = torch.zeros((self.node_counts[idx], 1))
        if self.edge_counts is not None:
            data.edge_index = torch.zeros((2, self.edge_counts[idx]))
        else:
            data.edge_index = torch.zeros((2, 0))
        return data

    def __len__(self):
        return len(self.node_counts)

def test_dynamic_batcher_with_node_counts():
    node_counts = [10, 20, 30, 40, 50]
    dataset = MockDataset(node_counts=node_counts)
    # Using batch_size=1 and collate_fn=lambda x: x[0] to mimic run_train.py
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    batcher = DynamicBatcher(
        loader=loader, max_num_nodes=60
    )

    batches = [b for b in batcher]
    # Batch 1: 10 + 20 + 30 = 60 <= 60.
    # Batch 2: 40. 40 + 50 > 60.
    # Batch 3: 50.
    assert len(batches) == 3
    assert [b.num_graphs for b in batches] == [3, 1, 1]
    assert batches[0].num_nodes == 60
    assert batches[1].num_nodes == 40
    assert batches[2].num_nodes == 50

def test_dynamic_batcher_with_edge_counts():
    edge_counts = [100, 200, 300, 400, 500]
    dataset = MockDataset(node_counts=[0] * 5, edge_counts=edge_counts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    batcher = DynamicBatcher(
        loader=loader, max_num_edges=600
    )

    batches = [b for b in batcher]
    assert len(batches) == 3
    assert [b.num_graphs for b in batches] == [3, 1, 1]
    assert batches[0].num_edges == 600

def test_dynamic_batcher_len_raises():
    node_counts = [10, 20]
    dataset = MockDataset(node_counts=node_counts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    batcher = DynamicBatcher(loader=loader)
    with pytest.raises(ValueError):
        len(batcher)

def test_dynamic_batcher_len_with_num_steps():
    node_counts = [10, 20]
    dataset = MockDataset(node_counts=node_counts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    batcher = DynamicBatcher(loader=loader, num_steps=10)
    assert len(batcher) == 10

def test_dynamic_batcher_skips_large():
    node_counts = [10, 70, 20]
    dataset = MockDataset(node_counts=node_counts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    batcher = DynamicBatcher(loader=loader, max_num_nodes=60)
    
    batches = [b for b in batcher]
    # 0: 10
    # 1: 70 > 60, skip
    # 2: 20
    assert len(batches) == 1
    assert [b.num_graphs for b in batches] == [2] # Wait, if 1 is skipped, it should be 10 and 20?
     
    # My implementation above appends to current batch.
    # [10]
    # 70 skipped.
    # [10, 20] fits in 60.
    assert batches[0].num_nodes == 30

def test_dynamic_batcher_respects_num_steps_early_exit():
    class LargeLoader:
        def __init__(self):
            self.count = 0
            
        def __iter__(self):
            self.count = 0
            # A large number of items
            for _ in range(100):
                self.count += 1
                mock_data = MagicMock()
                # Set attributes so they don't fail when accessed
                mock_data.num_nodes = 10
                mock_data.num_edges = 10
                # Loader yields list of data usually
                yield [mock_data]

    loader = LargeLoader()
    
    # We want to yield 2 batches.
    # max_num_nodes=15. Item size=10.
    # So 1 item per batch.
    # Should yield 2 batches, consuming 2 items.
    
    # We need to patch Batch because DynamicBatcher uses it to create return value
    with patch("mace.data.sampler.Batch") as MockBatch:
        MockBatch.from_data_list.side_effect = lambda x: f"batch_{len(x)}"
        
        batcher = DynamicBatcher(loader, max_num_nodes=15, num_steps=2)
        batches = list(batcher)
        
        assert len(batches) == 2
        # Without fix, it iterates the whole loader (100 times)
        # With fix, it should stop after yielding 2 batches (approx 2-3 iterations)
        assert loader.count < 10, f"Loader iterated {loader.count} times, expected < 10"
