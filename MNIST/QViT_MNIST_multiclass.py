#!/usr/bin/env python
# coding: utf-8

# ## Experiment
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from filelock import FileLock
import numpy as np
from jax.experimental import host_callback

from jax import config
config.update("jax_enable_x64", True)

# Check JAX backend (e.g., CPU or GPU)
print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# QViT Model Classes (Adapted for JAX)
class QSAL_pennylane:
    def __init__(self, S, n, Denc, D):
        self.seq_num = S  # Number of sequence positions (16 for 4x4 patches)
        self.num_q = n    # Number of qubits
        self.Denc = Denc  # Depth of encoding ansatz
        self.D = D        # Depth of Q, K, V ansatzes
        self.d = 49      # Dimension of input/output vectors (7x7 patches)
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
        input_flat = jnp.reshape(input, (-1, d))  # Flatten batch and sequence dimensions together

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

class QSANN_image_classifier:
    def __init__(self, S, n, Denc, D, num_layers, num_classes=10):
        self.Qnn = QSANN_pennylane(S, n, Denc, D, num_layers)
        self.d = n * (Denc + 2)
        self.S = S
        self.num_layers = num_layers
        self.num_classes = num_classes

    def __call__(self, x, params):
        # QNN
        qnn_params = params['qnn']
        x = self.Qnn(x, qnn_params)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        # Final layer
        w = params['final']['weight']
        b = params['final']['bias']
        logits = jnp.dot(x, w) + b
        # Apply softmax for multiclass classification
        return jax.nn.softmax(logits, axis=-1)

# Loss and Metrics
def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy loss"""
    # Clip predictions to avoid log(0)
    eps = 1e-7
    y_pred = jnp.clip(y_pred, eps, 1.0 - eps)
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=1))

def accuracy(y_true, y_pred):
    """Multiclass accuracy"""
    return jnp.mean(jnp.argmax(y_pred, axis=1) == jnp.argmax(y_true, axis=1))

# Evaluation Function
def evaluate(model, params, x, y):
    y_pred = model(x, params)
    loss = categorical_cross_entropy(y, y_pred)
    acc = accuracy(y, y_pred)
    return loss, acc

def create_patches(images, patch_size=7):
    """Convert MNIST images into patches.
    
    Args:
        images: Array of shape (batch_size, 28, 28)
        patch_size: Size of each square patch
    
    Returns:
        patches: Array of shape (batch_size, num_patches, patch_size*patch_size)
    """
    batch_size = images.shape[0]
    img_size = 28
    num_patches_per_dim = img_size // patch_size
    num_patches = num_patches_per_dim * num_patches_per_dim
    
    # Reshape to extract patches
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            # Extract patch
            patch = images[:, i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size]
            # Flatten patch
            patch = patch.reshape(batch_size, -1)
            patches.append(patch)
    
    # Stack patches
    patches = jnp.stack(patches, axis=1)  # Shape: (batch_size, num_patches, patch_size*patch_size)
    return patches

def load_mnist_data(n_train, n_test):
    """Load and preprocess MNIST dataset for multiclass classification."""
    # Load MNIST
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.reshape(-1, 28, 28)
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Convert to one-hot encoding
    y_onehot = jnp.zeros((y.size, 10))
    y_onehot = y_onehot.at[jnp.arange(y.size), y].set(1)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, train_size=n_train, test_size=n_test, stratify=y)
    
    # Create patches
    X_train_patches = create_patches(X_train)
    X_test_patches = create_patches(X_test)
    
    return X_train_patches, y_train, X_test_patches, y_test

# Parameter Initialization
def init_params(S, n, Denc, D, num_layers, num_classes=10):
    # Use a specific seed to match PyTorch
    key = jax.random.PRNGKey(42)
    d = 49  # Fixed dimension for 7x7 patches
    
    # Split key for different parameter groups
    keys = jax.random.split(key, num_layers * 3 + 2)
    
    # Initialize QNN parameters
    params = {
        'qnn': [
            {
                'Q': (jnp.pi / 4) * (2 * jax.random.normal(keys[i*3], (n * (D + 2),), dtype=jnp.float32) - 1),
                'K': (jnp.pi / 4) * (2 * jax.random.normal(keys[i*3 + 1], (n * (D + 2),), dtype=jnp.float32) - 1),
                'V': (jnp.pi / 4) * (2 * jax.random.normal(keys[i*3 + 2], (n * (D + 2),), dtype=jnp.float32) - 1)
            } for i in range(num_layers)
        ],
        'final': {
            'weight': 0.01 * jax.random.normal(keys[-2], (d * S, num_classes), dtype=jnp.float32),
            'bias': jnp.zeros((num_classes,), dtype=jnp.float32)
        }
    }
    return params

# Training Function
def train_qvit(n_train, n_test, n_epochs):
    # Load data
    x_train, y_train, x_test, y_test = load_mnist_data(n_train, n_test)

    # Initialize model and parameters (S=16 for 4x4 patches)
    model = QSANN_image_classifier(S=16, n=7, Denc=5, D=1, num_layers=1)
    params = init_params(S=16, n=7, Denc=5, D=1, num_layers=1)
    
    # Define optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)

    # Create arrays to store metrics
    metrics = {
        'train_loss': jnp.zeros(n_epochs),
        'train_acc': jnp.zeros(n_epochs),
        'test_loss': jnp.zeros(n_epochs),
        'test_acc': jnp.zeros(n_epochs)
    }

    # Loss function
    def loss_fn(p, x, y):
        y_pred = model(x, p)
        return categorical_cross_entropy(y, y_pred), y_pred

    # JIT-compiled update step
    @jax.jit
    def update_step(state, epoch):
        params, opt_state, metrics = state

        # Get both value and gradient, along with model predictions
        (loss_val, y_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_train, y_train)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        # Compute metrics
        train_acc = accuracy(y_train, y_pred)
        test_loss, test_acc = evaluate(model, new_params, x_test, y_test)

        # Update metrics arrays
        new_metrics = {
            'train_loss': metrics['train_loss'].at[epoch].set(loss_val),
            'train_acc': metrics['train_acc'].at[epoch].set(train_acc),
            'test_loss': metrics['test_loss'].at[epoch].set(test_loss),
            'test_acc': metrics['test_acc'].at[epoch].set(test_acc)
        }

        return (new_params, new_opt_state, new_metrics)

    # Training loop
    state = (params, opt_state, metrics)
    start = time.time()
    
    # Run training loop
    for epoch in range(n_epochs):
        state = update_step(state, epoch)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            params, _, metrics = state
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {metrics['train_loss'][epoch]:.4f} | "
                  f"Train Acc: {metrics['train_acc'][epoch]:.4f} | "
                  f"Test Loss: {metrics['test_loss'][epoch]:.4f} | "
                  f"Test Acc: {metrics['test_acc'][epoch]:.4f}")

    training_time = time.time() - start
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Save metrics to CSV after training
    final_metrics = state[2]
    df = pd.DataFrame({
        'epoch': range(1, n_epochs + 1),
        'train_loss': final_metrics['train_loss'],
        'train_acc': final_metrics['train_acc'],
        'test_loss': final_metrics['test_loss'],
        'test_acc': final_metrics['test_acc']
    })
    df.to_csv('training_log.csv', index=False)
    print("Metrics saved to training_log.csv")

    return state[0]  # Return final parameters

# Constants
n_test = 1000  # Increased test size for better evaluation
n_epochs = 100
n_train = 5000  # Increased train size for better learning

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