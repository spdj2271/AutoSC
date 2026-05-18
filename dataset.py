import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import cluster_metric


class SequenceDataset(Dataset):
    def __init__(self, X, assignments, len_sequence, fill_last=True, seed=1):
        """
        Dataset for training the autoregressive assignment Transformer.

        X:
            Input sample features, ordered by the assignment sequence.
            Shape: [N, D]

        assignments:
            Pseudo cluster assignments constructed in Stage 1.
            Shape: [N]

        len_sequence:
            Length of each training sequence.

        fill_last:
            If the last sequence has fewer than len_sequence samples,
            randomly sample existing samples to complete it.
        """
        assert len(X) == len(assignments), f"len(X)={len(X)} != len(assignments)={len(assignments)}"

        # Convert numpy arrays to torch tensors.
        # This allows the dataset to accept either numpy arrays or torch tensors.
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
            assignments = torch.from_numpy(assignments).long()

        self.len_sequence = int(len_sequence)
        self.fill_last = bool(fill_last)

        if self.len_sequence <= 0:
            raise ValueError(f"len_sequence must be positive, got {len_sequence}")

        n_total = len(X)

        if n_total <= 0:
            raise ValueError("Empty dataset is not allowed.")

        # We split the ordered samples into non-overlapping sequences.
        #
        # Example:
        #   len_sequence = 4
        #   samples = [x0, x1, x2, x3, x4, x5]
        #
        # Then the first sequence is:
        #   [x0, x1, x2, x3]
        #
        # The remaining [x4, x5] is shorter than len_sequence.
        # If fill_last=True, we randomly sample two existing samples to complete it.
        n_remainder = n_total % self.len_sequence
        n_missing = 0 if n_remainder == 0 else self.len_sequence - n_remainder

        self.original_len = n_total
        self.n_missing = n_missing if self.fill_last else 0

        if n_remainder != 0 and not self.fill_last:
            raise ValueError(
                f"Dataset length is not divisible by len_sequence: "
                f"len(X)={n_total}, len_sequence={self.len_sequence}. "
                f"Set fill_last=True to randomly complete the last sequence."
            )

        # Complete the last short sequence by randomly drawing samples from the dataset.
        # This avoids dropping samples and keeps every training sequence fixed-length.
        if self.n_missing > 0:
            rng = np.random.default_rng(seed)
            pad_indices = rng.choice(n_total, size=self.n_missing, replace=True)

            pad_indices = torch.from_numpy(pad_indices).long()
            self.X = torch.cat([X, X[pad_indices]], dim=0)
            self.assignments = torch.cat([assignments, assignments[pad_indices]], dim=0)
        else:
            self.X = X
            self.assignments = assignments

        # After padding, the total length should be divisible by len_sequence.
        self.total_len = len(self.X)
        self.n_sequences = self.total_len // self.len_sequence

        assert self.total_len % self.len_sequence == 0, (
            f"Internal error: total_len={self.total_len} is not divisible by "
            f"len_sequence={self.len_sequence}"
        )

    def __len__(self):
        # Number of fixed-length sequences.
        return self.n_sequences

    def __getitem__(self, idx):
        # Return one non-overlapping sequence.
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(f"idx out of range: idx={idx}, n_sequences={self.n_sequences}")

        start = idx * self.len_sequence
        end = start + self.len_sequence

        return self.X[start:end], self.assignments[start:end]
