#!/usr/bin/env python
# coding: utf-8

# ## Experiment

# In[97]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pennylane as qml
import time

# Check if CUDA (GPU) is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# QViT Model Classes (with fix for device handling)
class TorchLayer(nn.Module):
    def __init__(self, qnode, weights):
        if not torch.cuda.is_available():
            raise ImportError("TorchLayer requires PyTorch with CUDA support.")
        super().__init__()
        self.qnode = qnode
        self.qnode.interface = "torch"
        self.qnode_weights = {k: v for k, v in weights.items()}  # Move weights to device

    def forward(self, inputs):
        if len(inputs.shape) > 1:
            reconstructor = [self.forward(x) for x in torch.unbind(inputs)]
            return torch.stack(reconstructor)
        return self._evaluate_qnode(inputs)

    def _evaluate_qnode(self, x):
        kwargs = {
            self.input_arg: x,
            **{arg: weight for arg, weight in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)
        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)
        return torch.hstack(res).type(x.dtype)

    def __str__(self):
        return f"<Quantum Torch Layer: func={self.qnode.func.__name__}>"

    __repr__ = __str__
    _input_arg = "inputs"

    @property
    def input_arg(self):
        return self._input_arg

class QSAL_pennylane(nn.Module):
    def __init__(self, S, n, Denc, D):
        super().__init__()
        self.seq_num = S  # Number of sequence positions
        self.num_q = n    # Number of qubits
        self.Denc = Denc  # Depth of encoding ansatz
        self.D = D        # Depth of Q, K, V ansatzes
        self.d = n * (Denc + 2)  # Dimension of input/output vectors
        self.dev = qml.device("default.qubit", wires=self.num_q)

        # Define observables for value circuit (cycle through Z, X, Y per qubit)
        self.observables = []
        for i in range(self.d):
            qubit = i % self.num_q
            pauli_idx = (i // self.num_q) % 3
            if pauli_idx == 0:
                obs = qml.PauliZ(qubit)
            elif pauli_idx == 1:
                obs = qml.PauliX(qubit)
            else:
                obs = qml.PauliY(qubit)
            self.observables.append(obs)

        # Initialize single set of parameters for Q, K, V
        self.init_params_Q = nn.Parameter((np.pi / 4) * (2 * torch.randn(n * (D + 2)) - 1))
        self.init_params_K = nn.Parameter((np.pi / 4) * (2 * torch.randn(n * (D + 2)) - 1))
        self.init_params_V = nn.Parameter((np.pi / 4) * (2 * torch.randn(n * (D + 2)) - 1))

        # Weight dictionaries for TorchLayer
        self.weight_q = {"weights": self.init_params_Q}
        self.weight_k = {"weights": self.init_params_K}
        self.weight_v = {"weights": self.init_params_V}

        # Define quantum nodes
        self.vqnod = qml.QNode(self.circuit_v, self.dev, interface="torch")
        self.qnod = qml.QNode(self.circuit_qk, self.dev, interface="torch")

        # Single TorchLayer instances for Q, K, V
        self.q_linear = TorchLayer(self.qnod, self.weight_q)
        self.k_linear = TorchLayer(self.qnod, self.weight_k)
        self.v_linear = TorchLayer(self.vqnod, self.weight_v)

    def circuit_v(self, inputs, weights):
        """Value circuit returning a d-dimensional vector of different observable expectations."""
        idx = 0
        # Encoding ansatz U_enc
        for j in range(self.num_q):
            qml.RX(inputs[idx], wires=j)
            qml.RY(inputs[idx + 1], wires=j)
            idx += 2
        for i in range(self.Denc):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                qml.RY(inputs[idx], wires=j)
                idx += 1
        # Value ansatz U_v
        idx = 0
        for j in range(self.num_q):
            qml.RX(weights[idx], wires=j)
            qml.RY(weights[idx + 1], wires=j)
            idx += 2
        for i in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                qml.RY(weights[idx], wires=j)
                idx += 1
        return [qml.expval(obs) for obs in self.observables]

    def circuit_qk(self, inputs, weights):
        """Query/Key circuit returning Pauli-Z expectation on qubit 0."""
        idx = 0
        # Encoding ansatz U_enc
        for j in range(self.num_q):
            qml.RX(inputs[idx], wires=j)
            qml.RY(inputs[idx + 1], wires=j)
            idx += 2
        for i in range(self.Denc):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                qml.RY(inputs[idx], wires=j)
                idx += 1
        # Query/Key ansatz U_q or U_k
        idx = 0
        for j in range(self.num_q):
            qml.RX(weights[idx], wires=j)
            qml.RY(weights[idx + 1], wires=j)
            idx += 2
        for i in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                qml.RY(weights[idx], wires=j)
                idx += 1
        return [qml.expval(qml.PauliZ(0))]

    def forward(self, input):
        """Vectorized forward pass with shared parameters."""
        batch_size = input.size(0)
        S = self.seq_num
        d = self.d

        # Reshape input to (S * batch_size, d) for batch processing
        input_flat = input.transpose(0, 1).reshape(S * batch_size, d)

        # Compute Q, K, V in single calls
        Q_output_flat = self.q_linear(input_flat)  # Shape: (S * batch_size, 1)
        K_output_flat = self.k_linear(input_flat)  # Shape: (S * batch_size, 1)
        V_output_flat = self.v_linear(input_flat)  # Shape: (S * batch_size, d)

        # Reshape back to include sequence dimension
        Q_output = Q_output_flat.reshape(S, batch_size, 1).transpose(0, 1)  # (batch_size, S, 1)
        K_output = K_output_flat.reshape(S, batch_size, 1).transpose(0, 1)  # (batch_size, S, 1)
        V_output = V_output_flat.reshape(S, batch_size, d).transpose(0, 1)  # (batch_size, S, d)

        # Compute Gaussian self-attention coefficients
        Q_expanded = Q_output.unsqueeze(2)  # (batch_size, S, 1, 1)
        K_expanded = K_output.unsqueeze(1)  # (batch_size, 1, S, 1)
        alpha = torch.exp(-(Q_expanded - K_expanded) ** 2)  # (batch_size, S, S, 1)

        # Normalize attention coefficients
        Sum_a = torch.sum(alpha, dim=2, keepdim=True)  # (batch_size, S, 1, 1)
        alpha_normalized = alpha / Sum_a  # (batch_size, S, S, 1)

        # Compute weighted sum of values
        V_output_expanded = V_output.unsqueeze(1)  # (batch_size, 1, S, d)
        weighted_V = alpha_normalized * V_output_expanded  # (batch_size, S, S, d)
        Sum_w = torch.sum(weighted_V, dim=2)  # (batch_size, S, d)

        # Add residual connection
        output = input + Sum_w
        return output


class QSANN_pennylane(nn.Module):
    def __init__(self, S, n, Denc, D, num_layers):
        super().__init__()
        self.qsal_lst = [QSAL_pennylane(S, n, Denc, D) for _ in range(num_layers)]
        self.qnn = nn.Sequential(*self.qsal_lst)

    def forward(self, input):
        return self.qnn(input)

class QSANN_text_classifier(nn.Module):
    def __init__(self, S, n, Denc, D, num_layers):
        super().__init__()
        self.Qnn = QSANN_pennylane(S, n, Denc, D, num_layers)
        self.final_layer = nn.Linear(n * (Denc + 2) * S, 1).float()

    def forward(self, input):
        x = self.Qnn(input)
        x = torch.flatten(x, start_dim=1)
        return torch.sigmoid(self.final_layer(x))



# In[98]:

# Training and Evaluation Code
def load_digits_data(n_train, n_test):
    """
    Load and preprocess digits dataset for digits 0 and 1.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    mask = (y == 0) | (y == 1)  # Filter for digits 0 and 1
    X, y = X[mask], y[mask]
    X = X / np.linalg.norm(X, axis=1,keepdims=True)
    X = X.reshape(-1,4,16)
    y = y.astype(np.float32)  # Labels are already 0 and 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=42)
    print(y_train)
    # Convert to tensors and move to device
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    )

def train_qvit(n_train, n_test, n_epochs):
    """
    Train the QViT model and track metrics, ensuring all tensors are on the same device.
    """
    # Load data (already moved to device)
    x_train, y_train, x_test, y_test = load_digits_data(n_train, n_test)

    # Initialize QViT model and move it to the device
    model = QSANN_text_classifier(S=4, n=4, Denc=2, D=1, num_layers=1)

    # Define loss and optimizer
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.01)

    # Data containers
    train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []

    for epoch in range(n_epochs):
        start=time.time()
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # Record training metrics
        train_cost = loss.item()
        train_acc = ((outputs > 0.5).float() == y_train).float().mean().item()
        train_cost_epochs.append(train_cost)
        train_acc_epochs.append(train_acc)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_acc = ((test_outputs > 0.5).float() == y_test).float().mean().item()
        test_cost_epochs.append(test_loss)
        test_acc_epochs.append(test_acc)

        # Print progress every 10 epochs
        if (epoch + 1) % 1 == 0:
            print(f"Train Size: {n_train}, Epoch: {epoch + 1}/{n_epochs}, "
                  f"Train Loss: {train_cost:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f'Epoch time: {time.time()-start:.6f} seconds')

    return dict(
        status=["new"]*n_epochs,
        # n_train=[n_train] * n_epochs,
        step=np.arange(1, n_epochs + 1, dtype=int),
        train_cost=train_cost_epochs,
        train_acc=train_acc_epochs,
        test_cost=test_cost_epochs,
        test_acc=test_acc_epochs,
    )

# Constants
n_test = 100
n_epochs = 100
n_reps = 1
train_sizes = [5]

def run_iterations(n_train):
    """
    Run multiple training iterations for a given training size and print progress.
    """
    results_df = pd.DataFrame(
        columns=["status","train_acc", "train_cost", "test_acc", "test_cost", "step"]
    )
    for rep in range(n_reps):
        print(f"\nStarting repetition {rep + 1}/{n_reps} for train size {n_train}")
        results = train_qvit(n_train, n_test, n_epochs)
        results_df = pd.concat([results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True)
    return results_df


# In[ ]:


import pandas as pd
from filelock import FileLock

output_file = "results.csv"
lock_file = output_file + ".lock"

# Create an empty file with header once (only if it doesn't exist)
try:
    with open(output_file, 'x') as f:
        pd.DataFrame(columns=["status","train_acc", "train_cost", "test_acc", "test_cost", "step"]).to_csv(f, index=False)
except FileExistsError:
    pass  # File already exists

for n_train in train_sizes:
    print(f"\n=== Starting training for train size {n_train} ===")
    results_df = run_iterations(n_train)

    # Append safely to CSV with file locking
    with FileLock(lock_file):
        results_df.to_csv(output_file, mode='a', index=False, header=False)

