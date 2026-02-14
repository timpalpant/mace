import torch
from torch.utils.data import DataLoader
from typing import Iterator, List, Optional
from mace.tools.torch_geometric.batch import Batch

class DynamicBatcher:
    def __init__(
        self,
        loader: DataLoader,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
        num_steps: Optional[int] = None,
    ):
        self.loader = loader
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.num_steps = num_steps

    def __iter__(self) -> Iterator[Batch]:
        batch = []
        curr_nodes = 0
        curr_edges = 0
        num_yielded = 0
        
        for data_batch in self.loader:
            if self.num_steps is not None and num_yielded >= self.num_steps:
                break

            if not isinstance(data_batch, list):
                data_batch = [data_batch]
                
            for data in data_batch:
                if self.num_steps is not None and num_yielded >= self.num_steps:
                    break
                    
                num_nodes = data.num_nodes
                num_edges = data.num_edges
                
                # Check if this single item is too large (already handled by sampler usually, but safe here)
                if (self.max_num_nodes and num_nodes > self.max_num_nodes) or \
                   (self.max_num_edges and num_edges > self.max_num_edges):
                    continue

                if (self.max_num_nodes and curr_nodes + num_nodes > self.max_num_nodes) or \
                   (self.max_num_edges and curr_edges + num_edges > self.max_num_edges):
                    if batch:
                        yield Batch.from_data_list(batch)
                        num_yielded += 1
                        batch = []
                        curr_nodes = 0
                        curr_edges = 0
                    
                    if self.num_steps is not None and num_yielded >= self.num_steps:
                        break
                
                batch.append(data)
                curr_nodes += num_nodes
                curr_edges += num_edges
            
        if batch and (self.num_steps is None or num_yielded < self.num_steps):
            yield Batch.from_data_list(batch)

    def __len__(self) -> int:
        if self.num_steps is not None:
            return self.num_steps
        raise ValueError(
            "The length of 'DynamicBatcher' is undefined since the number of batches "
            "depends on the sample sizes. Specify 'num_steps' if you need the length for progress bars."
        )
