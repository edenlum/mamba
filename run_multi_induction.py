from asyncio import Event

from tqdm import tqdm
from typing import Tuple
from dataclasses import dataclass

from ray.actor import ActorHandle


import ray
import os

import traceback


os.environ["WANDB_SILENT"] = "true"



@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int = 1) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    """
    Usage example:
    >> pb = ProgressBar()
    >> futures = [somefunc.remote(someinput, pb.actor) for someinput in someinputs]
    >> pb.set_total(len(futures))
    >> pb.print_until_done()
    """
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.description = description

    def set_total(self, total: int):
        self.total = total

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total,
                    smoothing=0)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


# Assumptions: 'model', 'dataloader', 'device', 'optim' (optimizer) are already defined
def train(config, model, data_loader, optimizer):
    import torch.nn.functional as F
    import wandb
    import torch
    import math
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Setup tqdm for the outer loop
    # pbar = tqdm(total=config.epochs, desc="Epoch Progress", position=0)

    # Training Loop
    for epoch in range(config.epochs):
        avg_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        total_correct_sequences = 0
        first_token_correct_count = 0
        last_token_correct_count = 0
        for data, labels in data_loader:
            data = data.to(device).long()  # Ensure data is on the correct device and dtype
            labels = labels.to(device).long()  # Ensure labels are on the correct device and converted to long

            # Forward pass
            logits = model(data)  # [batch_size, seq_len, cat_num]

            # Compute loss
            loss = F.cross_entropy(logits[:, -config.induction_len:, :].reshape(-1, config.n_categories),
                                   labels[:, -config.induction_len:].reshape(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            # Calculate predictions
            _, predicted = torch.max(logits, dim=2)  # [batch_size, seq_len]

            # Mask to focus only on relevant positions
            relevant_labels = labels[:, -config.induction_len:]
            relevant_predicted = predicted[:, -config.induction_len:]

            # Calculate correct predictions per token
            correct_tokens = (relevant_predicted == relevant_labels).sum()
            total_correct_tokens += correct_tokens.item()
            total_tokens += relevant_labels.numel()  # Total number of evaluated tokens

            # Calculate correct predictions per sequence
            correct_sequences = (relevant_predicted == relevant_labels).all(dim=1).sum()
            total_correct_sequences += correct_sequences.item()

            # Accuracy for the first and last tokens in the sequence
            first_token_correct_count += (relevant_predicted[:, 0] == relevant_labels[:, 0]).sum().item()
            last_token_correct_count += (relevant_predicted[:, -1] == relevant_labels[:, -1]).sum().item()

        total_sequences = sum(len(labels) for _, labels in data_loader)
        avg_loss /= len(data_loader)
        avg_accuracy_per_token = total_correct_tokens / total_tokens
        avg_accuracy_per_sequence = total_correct_sequences / total_sequences
        first_token_accuracy = first_token_correct_count / total_sequences
        last_token_accuracy = last_token_correct_count / total_sequences

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "avg_accuracy_per_token": avg_accuracy_per_token,
            "avg_accuracy_per_sequence": avg_accuracy_per_sequence,
            "first_token_accuracy": first_token_accuracy,
            "last_token_accuracy": last_token_accuracy
        })

        if config.stop_on_loss:
            if avg_loss < config.stop_on_loss or math.isnan(avg_loss):
                break

    # pbar.close()


@dataclass
class Config:
    ssm_type: str
    initA_real: str
    initA_imag: str
    discretizationA: str
    discretizationB: str
    param_A_imag: str
    A_imag_using_weight_decay: str
    dt_is_selective: bool
    channel_sharing: str
    deterministic: bool
    pscan: bool
    d_model: int
    d_state: int
    n_layers: int
    n_categories: int
    induction_len: int
    num_triggers: int
    seq_len: int
    batch_size: int
    epoch_size: int
    epochs: int
    lr: float
    stop_on_loss: float
    seed: int
    comment: str
    bias: bool
    use_cuda:bool


def experiments(kwargs):
    import itertools
    # Extract argument names and their corresponding value lists
    arg_names = [k[0] for k in kwargs]
    value_lists = [k[1] for k in kwargs]

    # Iterate over the Cartesian product of value lists
    for values in itertools.product(*value_lists):
        # Yield a dictionary mapping argument names to values
        yield dict(zip(arg_names, values))

@ray.remote(num_gpus=0.25)# if torch.cuda.is_available() else 0)
def run_experiment(config,progress_bar_actor):
    import wandb
    import torch
    from torch import optim
    from ds.datasets import InductionHead
    from simple_mamba.mamba_lm import MambaLM, MambaLMConfig
    import numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        exp_name = name(config)

        wandb.init(
            project="InductionS6complex",
            entity="yuv-milo",
            name=exp_name,
            config=config
        )
        wandb.log({"epoch": 1})
        wandb.log({"epoch": 2})

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        mamba_config = MambaLMConfig(
            ssm_type=config.ssm_type,
            discretizationA=config.discretizationA,
            discretizationB=config.discretizationB,
            initA_imag=config.initA_imag,
            initA_real=config.initA_real,
            param_A_imag = config.param_A_imag,
            A_imag_using_weight_decay=config.A_imag_using_weight_decay,
            dt_is_selective = config.dt_is_selective,
            channel_sharing = config.channel_sharing,
            d_model=config.d_model,
            d_state=config.d_state,
            n_layers=config.n_layers,
            vocab_size=config.n_categories,
            pad_vocab_size_multiple=config.n_categories,
            deterministic = config.deterministic,
            bias=config.bias,
            pscan = config.pscan,
            use_cuda=config.use_cuda
        )

        dataset = InductionHead(config.epoch_size,
                                config.seq_len,
                                config.n_categories,
                                config.num_triggers,
                                config.induction_len)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=config.batch_size,
                                                  shuffle=True)
        model = MambaLM(mamba_config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        train(config, model, data_loader, optimizer)
    except Exception as e:
        print(progress_bar_actor, "fail:", traceback.format_exc())
    progress_bar_actor.update.remote()
    wandb.finish()

def name(config):
    # short name for display on wandb
    return f"{config.ssm_type}-d_state{config.d_state}-seq_len{config.seq_len}-ind_len{config.induction_len}"

def main():
    ray.init(num_cpus=32, ignore_reinit_error=True)
    pb = ProgressBar()
    progress_bar_actor = pb.actor

    induction_lens = [200, ]
    seq_len = [400, ]
    seeds = [1, 2]
    epochs = 10000
    n_categories = [16, 64]
    batch_size = 8

    settings_options = [
        ["seed", seeds],
        ["n_categories", n_categories],
        ["d_state", [16,]],
        ["induction_len", induction_lens],
        ["d_model", [64]],
        ["seq_len", seq_len],
        ["num_triggers", [1]],
        ["ssm_type", ["S4D-Real", "S4D-Complex"]],
        ["n_layers", [2]],
        ["batch_size", [batch_size]],
        ["epochs", [epochs]],  # [int(1600 * 6]],
        ["epoch_size", [256 * 6]],
        ["lr", [1e-3]],
        ["stop_on_loss", [0.01]],
        ["A_imag_using_weight_decay", ["True"]],
        ["initA_imag", ["S4"]],
        ["param_A_imag", ["normal", ]],
        ["discretizationB", ["s6"]],
        ["discretizationA", ["normal"]],
        ["initA_real", ["S6"]],
        ["dt_is_selective", [False]],
        ["channel_sharing", [False]],
        ["bias", [True]],
        ["deterministic", [False]],
        ["pscan", [True]],
        ["use_cuda", [True, ]],
    ]

    settings_options_s6_real = [
        ["seed", [2, 3]],
        ["num_triggers", [1]],
        ["ssm_type", ["S6-Real"]],
        ["discretizationA", ["normal"]],
        ["discretizationB", ["s6"]],
        ["d_model", [64]],
        ["d_state", [16]],
        ["induction_len", induction_lens],
        ["n_layers", [2]],
        ["n_categories", n_categories],
        ["batch_size", [batch_size]],
        ["epochs", [epochs]],  # [int(1600 * 6]],
        ["epoch_size", [128 * 4]],
        ["lr", [1e-3]],
        ["stop_on_loss", [0.01]],
        ["initA_imag", [None, ]],
        ["initA_real", [None, ]],
        ["param_A_imag", [None, ]],
        ["A_imag_using_weight_decay", [None, ]],
        ["dt_is_selective", [True, False]],
        ["channel_sharing", [True]],
        ["bias", [True]],
        ["deterministic", [False]],
        ["pscan", [False, ]],
        ["use_cuda", [True, ]],
        ["seq_len", seq_len],
    ]

    settings_options_s6_complex = [
        ["seed", [2, 3]],
        ["num_triggers", [1]],
        ["ssm_type", ["S6-Complex"]],
        ["discretizationA", ["normal"]],
        ["d_model", [64]],
        ["induction_len", induction_lens],
        ["n_layers", [2]],
        ["n_categories", n_categories],
        ["batch_size", [batch_size]],
        ["epochs", [epochs]],  # [int(1600 * 6]],
        ["epoch_size", [128 * 4]],
        ["lr", [1e-3]],
        ["stop_on_loss", [0.01]],
        ["param_A_imag", ["normal", ]],
        ["A_imag_using_weight_decay", ["True", ]],
        ["deterministic", [False]],
        ["pscan", [True]],
        ["bias", [True]],
        ["initA_imag", ["S4", ]],
        ["initA_real", ["S4", ]],
        ["dt_is_selective", [True, False]],
        ["discretizationB", ["s6"]],
        ["d_state", [8]],
        ["channel_sharing", [True]],
        ["pscan", [False, ]],
        ["use_cuda", [True, ]],
        ["seq_len", seq_len],
    ]

    tasks = []
    for i, config in enumerate(experiments(settings_options_s6_complex)):
        print(i)
        config.update({"comment": ""})
        tasks.append(run_experiment.remote(Config(**config), progress_bar_actor))
    for i, config in enumerate(experiments(settings_options_s6_real)):
        print(i)
        config.update({"comment": ""})
        tasks.append(run_experiment.remote(Config(**config), progress_bar_actor))
    for i, config in enumerate(experiments(settings_options)):
        print(i)
        config.update({"comment": ""})
        tasks.append(run_experiment.remote(Config(**config), progress_bar_actor))
    pb.set_total(len(tasks))
    pb.print_until_done()
    print("finished running all")
if __name__ == '__main__':
    main()