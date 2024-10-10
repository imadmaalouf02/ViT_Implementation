"""Sets random sets for torch operations."""

import torch


class set_seed:

    def __init__(self, seed: int=42):
        self.seed = seed

    def set_seeds(self):
            # Set the seed for general torch operations
            torch.manual_seed(self.seed)
            # Set the seed for CUDA torch operations (ones that happen on the GPU)
            torch.cuda.manual_seed(self.seed)