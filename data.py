import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


class BanditDataset(Dataset):
    def __init__(self, A, history):
        self.A = A
        self.k, self.d = A.shape

        history_map = defaultdict(list)
        for m, a, y in history:
            history_map[a].append(y)
        self.history = [
            (a, np.mean(ys), len(ys)) for a, ys in history_map.items()
        ]

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        a, y_bar, n = self.history[idx]
        return (
            torch.from_numpy(self.A[a, :]).float(),
            torch.tensor([y_bar]),
            torch.tensor([n]),
        )


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    A = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ])
    history = [
        (0, 1.0),
        (3, 2.0),
        (0, 2.0),
        (1, 0.3),
        (2, 0.1),
    ]
    dataset = BanditDataset(A, history)

    train_data = DataLoader(dataset, batch_size=2, shuffle=True)

    for X, y, n in train_data:
        print(X, y, n)

