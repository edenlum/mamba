import torch
from torch.utils.data import Dataset
import numpy as np

def delay_l2(lag):
    def delay_loss_func(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean((output[:, lag:, :] - target[:, lag:, :]) ** 2)
        return loss

    return delay_loss_func


class NormalNoiseSignalGenerator():
    def __init__(self, std=1, mean=0):
        self.std = std
        self.mean = mean

    def generate(self, num_signals, signal_length, signal_dim=1):
        if signal_dim != 1:
            raise NotImplementedError("ConstSignalGenerator only supports signal dim 1")

        signals = torch.randn([num_signals, signal_length])
        return signals


class DelayedSignalDatasetRegenerated(torch.utils.data.TensorDataset):

    def __init__(self, samples_num=1, seq_length=10000, lag_length=1000,
                 signal_generator=None, lag_type="zero"):
        assert lag_length < seq_length
        assert lag_type in ["zero"]

        if signal_generator is None:
            signal_generator = NormalNoiseSignalGenerator()
            # raise ValueError("signal_generator must be provided")

        self.signal_generator = signal_generator
        self.samples_num = samples_num
        self.seq_length = seq_length
        self.lag_length = lag_length
        self.lag_type = lag_type
        super().__init__()

    def __getitem__(self, index):
        X = self.signal_generator.generate(num_signals=1,
                                           signal_length=self.seq_length)
        X = X.unsqueeze(-1)

        if self.lag_type == "zero":
            Y = torch.zeros(X.shape)
            Y[:, self.lag_length:, :] = X[:, :-self.lag_length, :]

        return X[0, :, :], Y[0, :, :]

    def __len__(self):
        return self.samples_num
    

def logits_to_probability(logits):
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)
    # Convert the PyTorch tensor to a NumPy array
    probabilities_numpy = probabilities.detach().numpy()
    return probabilities_numpy

class DynamicCategoricalDataset(Dataset):
    def __init__(self, amount_of_examples, seq_len, cat_num, lag):
        """
        Initialize the dataset with the parameters for data generation.
        Args:
            amount_of_examples (int): The total number of examples (data points).
            seq_len (int): The length of each sequence.
            cat_num (int): The number of categories for each element in the sequence.
            lag (int): The number of steps to shift the data for label generation.
        """
        self.amount_of_examples = amount_of_examples
        self.seq_len = seq_len
        self.cat_num = cat_num
        self.lag = lag

    def __len__(self):
        """
        Return the total number of examples you want the loader to simulate.
        """
        return self.amount_of_examples

    def __getitem__(self, idx):
        """
        Generates a single data point on demand.
        """
        data = np.random.randint(0, self.cat_num, size=(self.seq_len,))
        labels = np.zeros_like(data)
        if self.lag < self.seq_len:
            labels[self.lag:] = data[:-self.lag]
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DynamicRepeatDataset(Dataset):
    def __init__(self, amount_of_examples, cat_num, lag, auto_regressive):
        """
        Initialize the dataset with the parameters for data generation.
        Args:
            amount_of_examples (int): The total number of examples (data points).
            seq_len (int): The length of each sequence.
            cat_num (int): The number of categories for each element in the sequence.
            lag (int): The number of steps to shift the data for label generation.
        """
        self.amount_of_examples = amount_of_examples
        self.cat_num = cat_num
        self.lag = lag
        self.auto_regressive = auto_regressive

    def __len__(self):
        """
        Return the total number of examples you want the loader to simulate.
        """
        return self.amount_of_examples

    def __getitem__(self, idx):
        """
        Generates a single data point on demand.
        """
        seq_len = 2 * self.lag - 1
        data = np.random.randint(0, self.cat_num, size=(seq_len,))
        if self.auto_regressive:
            data[-self.lag + 1:] = data[:self.lag - 1]
        labels = np.zeros_like(data)
        labels[self.lag:] = data[1:-self.lag + 1]
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class InductionHead(Dataset):
    def __init__(self, amount_of_examples, seq_len, cat_num, num_triggers, induction_length, copy_at_end = True):
        """
        Initialize the Induction Head dataset with the parameters for data generation.
        Args:
            amount_of_examples (int): The total number of examples (data points).
            seq_len (int): The length of each sequence.
            cat_num (int): The number of categories for each element in the sequence.
            num_triggers (int): The (maximum) number of times the copy token is shown in the sequence.
            induction_length (int): The length of the induction sequence (after the copy token).
        """
        self.amount_of_examples = amount_of_examples
        self.seq_len = seq_len
        self.cat_num = cat_num
        self.num_triggers = num_triggers
        self.induction_length = induction_length
        self.copy_token = 0
        self.copy_at_end = copy_at_end

    def __len__(self):
        """
        Return the total number of examples you want the loader to simulate.
        """
        return self.amount_of_examples

    def __getitem__(self, idx):
        """
        Generates a single data point on demand.
        """
        data = np.random.randint(1, self.cat_num, size=(self.seq_len,))
        if self.copy_at_end:
            data = np.append(data, 0)
        copy_indices = np.random.choice(np.arange(0, self.seq_len - (1 + self.induction_length)), size=self.num_triggers, replace=False)
        copy_indices_filtered = []
        for i, p in enumerate(copy_indices):
            if i == 0:
                copy_indices_filtered.append(p)
            elif p - copy_indices_filtered[-1] > self.induction_length:
                copy_indices_filtered.append(p)
        to_copy = [
            data[copy_indices_filtered[0]+1+i]
            for i in range(self.induction_length)
        ]
        for pos in copy_indices_filtered:
            data[pos] = self.copy_token
            for i in range(self.induction_length):
                data[pos+1+i] = to_copy[i]
        data = np.concatenate((data, to_copy))
        data, labels = data[:-1], data[1:]
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    