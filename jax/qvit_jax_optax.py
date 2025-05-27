#!/usr/bin/env python
# coding: utf-8

# ## Experiment
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from filelock import FileLock

from jax import config
config.update("jax_enable_x64", True)

# Check JAX backend (e.g., CPU or GPU)
print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# QViT Model Classes (Adapted for JAX)
class QSAL_pennylane:
    def __init__(self, S, n, Denc, D):
        self.seq_num = S  # Number of sequence positions
        self.num_q = n    # Number of qubits
        self.Denc = Denc  # Depth of encoding ansatz
        self.D = D        # Depth of Q, K, V ansatzes
        self.d = n * (Denc + 2)  # Dimension of input/output vectors
        self.dev = qml.device("default.qubit", wires=self.num_q)

        # Define observables for value circuit
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

        # Define quantum nodes with JAX interface
        self.vqnod = qml.QNode(self.circuit_v, self.dev, interface="jax")
        self.qnod = qml.QNode(self.circuit_qk, self.dev, interface="jax")

    def circuit_v(self, inputs, weights):
        """Value circuit returning a d-dimensional vector of observable expectations."""
        idx = 0
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

    def __call__(self, input, params_Q, params_K, params_V):
        batch_size = input.shape[0]
        S = self.seq_num
        d = self.d

        # Reshape input to (S * batch_size, d) for vectorized computation
        input_flat = input.reshape(S * batch_size, d)

        # Compute Q, K, V using vectorized operations
        Q_output_flat = jnp.array(jax.vmap(lambda x: self.qnod(x, params_Q))(input_flat)).T
        K_output_flat = jnp.array(jax.vmap(lambda x: self.qnod(x, params_K))(input_flat)).T
        V_output_flat = jnp.array(jax.vmap(lambda x: self.vqnod(x, params_V))(input_flat)).T

        # Reshape back to include sequence dimension
        Q_output = Q_output_flat.reshape(batch_size, S, 1)
        K_output = K_output_flat.reshape(batch_size, S, 1)
        V_output = V_output_flat.reshape(batch_size, S, d)

        # Compute Gaussian self-attention coefficients
        Q_expanded = Q_output[:, :, None, :]
        K_expanded = K_output[:, None, :, :]
        alpha = jnp.exp(-(Q_expanded - K_expanded) ** 2)
        Sum_a = jnp.sum(alpha, axis=2, keepdims=True)
        alpha_normalized = alpha / Sum_a

        # Compute weighted sum of values
        V_output_expanded = V_output[:, None, :, :]
        weighted_V = alpha_normalized * V_output_expanded
        Sum_w = jnp.sum(weighted_V, axis=2)

        # Add residual connection
        output = input + Sum_w
        return output

class QSANN_pennylane:
    def __init__(self, S, n, Denc, D, num_layers):
        self.qsal_lst = [QSAL_pennylane(S, n, Denc, D) for _ in range(num_layers)]

    def __call__(self, input, params):
        x = input
        for qsal, p in zip(self.qsal_lst, params):
            x = qsal(x, p['Q'], p['K'], p['V'])
        return x

class QSANN_text_classifier:
    def __init__(self, S, n, Denc, D, num_layers):
        self.Qnn = QSANN_pennylane(S, n, Denc, D, num_layers)
        self.d = n * (Denc + 2)
        self.S = S
        self.num_layers = num_layers

    def __call__(self, x, params):
        # Layer norm 1
        # x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5)
        # QNN
        qnn_params = params['qnn']
        x = self.Qnn(x, qnn_params)
        # Layer norm 2
        # x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        # Final layer
        w = params['final']['weight']
        b = params['final']['bias']
        logits = jnp.dot(x, w) + b
        return jax.nn.sigmoid(logits)

# Loss and Metrics
def binary_cross_entropy(y_true, y_pred):
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def accuracy(y_true, y_pred):
    return jnp.mean((y_pred > 0.5) == y_true)

# Evaluation Function
def evaluate(model, params, x, y):
    y_pred = model(x, params)
    loss = binary_cross_entropy(y, y_pred)
    acc = accuracy(y, y_pred)
    return loss, acc

# Data Loading
def load_digits_data(n_train, n_test):
    digits = load_digits()
    X, y = digits.data, digits.target
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    X = X / 16.0  # Normalize to [0, 1]
    X = X.reshape(-1, 4, 16)
    y = y.astype(jnp.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train, test_size=n_test)
    return (
        jnp.array(X_train),
        jnp.array(y_train).reshape(-1, 1),
        jnp.array(X_test),
        jnp.array(y_test).reshape(-1, 1)
    )

# Parameter Initialization
def init_params(S, n, Denc, D, num_layers):
    key = jax.random.PRNGKey(0)
    d = n * (Denc + 2)
    params = {
        'qnn': [
            {
                'Q': jax.random.uniform(key, (n * (D + 2),), minval=-jnp.pi/4, maxval=jnp.pi/4),
                'K': jax.random.uniform(key, (n * (D + 2),), minval=-jnp.pi/4, maxval=jnp.pi/4),
                'V': jax.random.uniform(key, (n * (D + 2),), minval=-jnp.pi/4, maxval=jnp.pi/4)
            } for _ in range(num_layers)
        ],
        'final': {
            'weight': jax.random.normal(key, (d * S, 1)),
            'bias': jax.random.normal(key, (1,))
        }
    }
    return params

# Training Function
def train_qvit(n_train, n_test, n_epochs):
    # Load data
    x_train, y_train, x_test, y_test = load_digits_data(n_train, n_test)

    # Initialize model and parameters
    model = QSANN_text_classifier(S=4, n=4, Denc=2, D=1, num_layers=1)
    params = init_params(S=4, n=4, Denc=2, D=1, num_layers=1)
    print(params['final'])
    # Define optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)

    # Loss function
    def loss_fn(p, x, y):
        y_pred = model(x, p)
        return binary_cross_entropy(y, y_pred)

    # Accuracy + evaluation
    def acc_fn(p):
        return accuracy(y_train, model(x_train, p)), evaluate(model, p, x_test, y_test)

    # JIT-compiled update step with debug printing
    @jax.jit
    def update_step(epoch, args):
        params, opt_state, x, y, print_training = args

        loss_val, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Use jax.debug.print (only prints on device)
        def print_fn():
            train_acc, (test_loss, test_acc) = acc_fn(params)
            jax.debug.print(
                "Epoch: {epoch} | Train Loss: {loss_val} | Train Acc: {train_acc} | "
                "Test Loss: {test_loss} | Test Acc: {test_acc}",
                epoch=epoch + 1,
                loss_val=loss_val,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
            )


        jax.lax.cond(print_training, print_fn, lambda: None)

        return (params, opt_state, x, y, print_training)

    # JIT-compiled outer loop
    @jax.jit
    def training_loop(params, opt_state, x, y, print_training):
        args = (params, opt_state, x, y, print_training)
        final_args = jax.lax.fori_loop(0, n_epochs, update_step, args)
        return final_args[0]  # return only the final parameters

    # Run training with debug print every 5 epochs
    start=time.time()
    final_params = training_loop(params, opt_state, x_train, y_train, True)
    print("Training time:", time.time() - start)

    return final_params

# Constants
n_test = 100
n_epochs = 100
n_reps = 1
n_train = 80

results = train_qvit(n_train, n_test, n_epochs)


# def run_iterations(n_train):
#     results_df = pd.DataFrame(
#         columns=["status", "train_acc", "train_cost", "test_acc", "test_cost", "step"]
#     )
#     for rep in range(n_reps):
#         print(f"\nStarting repetition {rep + 1}/{n_reps} for train size {n_train}")
#         results = train_qvit(n_train, n_test, n_epochs)
#         results_df = pd.concat([results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True)
#     return results_df

# # Output Handling
# output_file = "results.csv"
# lock_file = output_file + ".lock"

# try:
#     with open(output_file, 'x') as f:
#         pd.DataFrame(columns=["status", "train_acc", "train_cost", "test_acc", "test_cost", "step"]).to_csv(f, index=False)
# except FileExistsError:
#     pass  # File already exists

# for n_train in train_sizes:
#     print(f"\n=== Starting training for train size {n_train} ===")
#     results_df = run_iterations(n_train)
#     with FileLock(lock_file):
#         results_df.to_csv(output_file, mode='a', index=False, header=False)