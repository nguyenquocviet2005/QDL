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
import tensorflow as tf  # For loading CIFAR-10
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-GUI)
import matplotlib.pyplot as plt
import matplotlib as mpl

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
        self.d = 48      # Dimension of input/output vectors (8x8x3 patches)
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
        
        # x = (input - input.mean(axis=-1, keepdims=True)) / (input.std(axis=-1, keepdims=True) + 1e-5)
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
    """Binary cross-entropy loss matching PyTorch's nn.BCELoss behavior"""
    # Clip predictions to avoid log(0)
    eps = 1e-7
    y_pred = jnp.clip(y_pred, eps, 1.0 - eps)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def accuracy(y_true, y_pred):
    return jnp.mean((y_pred > 0.5) == y_true)

# Evaluation Function
def evaluate(model, params, x, y):
    y_pred = model(x, params)
    loss = binary_cross_entropy(y, y_pred)
    acc = accuracy(y, y_pred)
    return loss, acc

def create_patches(images, patch_size=4):
    """Convert CIFAR images into patches.
    
    Args:
        images: Array of shape (batch_size, 32, 32, 3)
        patch_size: Size of each square patch
    
    Returns:
        patches: Array of shape (batch_size, num_patches, patch_size*patch_size*3)
    """
    batch_size = images.shape[0]
    img_size = 32
    num_patches_per_dim = img_size // patch_size
    num_patches = num_patches_per_dim * num_patches_per_dim
    
    # Reshape to extract patches
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            # Extract patch (including all color channels)
            patch = images[:, i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size, :]
            # Flatten patch (8x8x3 = 192 dimensions)
            patch = patch.reshape(batch_size, -1)
            patches.append(patch)
    
    # Stack patches
    patches = jnp.stack(patches, axis=1)  # Shape: (batch_size, num_patches, patch_size*patch_size*3)
    return patches

def augment_image(image):
    """Apply random data augmentation to a single image (TensorFlow tensor)."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image

def load_cifar_data(n_train, n_test, binary=True, augment=True):
    """Load and preprocess CIFAR-10 dataset with optional data augmentation."""
    # Load CIFAR-10
    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
    
    if binary:
        # Use only two classes (0: airplane, 1: automobile)
        mask_train = (y_train_full[:, 0] == 0) | (y_train_full[:, 0] == 1)
        mask_test = (y_test_full[:, 0] == 0) | (y_test_full[:, 0] == 1)
        X_train_full = X_train_full[mask_train]
        y_train_full = y_train_full[mask_train]
        X_test_full = X_test_full[mask_test]
        y_test_full = y_test_full[mask_test]
        # Convert labels to binary (0 or 1)
        y_train_full = (y_train_full == 1).astype(float)
        y_test_full = (y_test_full == 1).astype(float)

    # Normalize pixel values to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test_full = X_test_full.astype('float32') / 255.0

    # Select subset of data
    indices_train = np.random.choice(len(X_train_full), n_train, replace=False)
    indices_test = np.random.choice(len(X_test_full), n_test, replace=False)
    X_train = X_train_full[indices_train]
    y_train = y_train_full[indices_train]
    X_test = X_test_full[indices_test]
    y_test = y_test_full[indices_test]

    # Data augmentation (only for training set)
    if augment:
        X_train_tf = tf.convert_to_tensor(X_train)
        X_train_tf = tf.map_fn(augment_image, X_train_tf)
        X_train = X_train_tf.numpy()

    # Create patches
    X_train_patches = create_patches(X_train)
    X_test_patches = create_patches(X_test)
    
    return (
        jnp.array(X_train_patches),
        jnp.array(y_train),
        jnp.array(X_test_patches),
        jnp.array(y_test)
    )

# Parameter Initialization
def init_params(S, n, Denc, D, num_layers):
    # Use a specific seed to match PyTorch
    key = jax.random.PRNGKey(42)
    d = 48  # Fixed dimension for 8x8x3 patches
    
    # Split key for different parameter groups
    keys = jax.random.split(key, num_layers * 3 + 2)
    
    # Initialize QNN parameters with std=1.0 to match torch.randn
    params = {
        'qnn': [
            {
                'Q': (jnp.pi / 4) * (2 * jax.random.normal(keys[i*3], (n * (D + 2),), dtype=jnp.float32) - 1),
                'K': (jnp.pi / 4) * (2 * jax.random.normal(keys[i*3 + 1], (n * (D + 2),), dtype=jnp.float32) - 1),
                'V': (jnp.pi / 4) * (2 * jax.random.normal(keys[i*3 + 2], (n * (D + 2),), dtype=jnp.float32) - 1)
            } for i in range(num_layers)
        ],
        'final': {
            'weight': 0.01 * jax.random.normal(keys[-2], (d * S, 1), dtype=jnp.float32),
            'bias': jnp.zeros((1,), dtype=jnp.float32)
        }
    }
    return params

# Training Function
def train_qvit(n_train, n_test, n_epochs):
    # Load data
    x_train, y_train, x_test, y_test = load_cifar_data(n_train, n_test)

    # Initialize model and parameters
    model = QSANN_image_classifier(S=64, n=4, Denc=2, D=1, num_layers=1)
    params = init_params(S=64, n=4, Denc=2, D=1, num_layers=1)
    
    # Define optimizer
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    # Create arrays to store metrics
    train_costs = []
    test_costs = []
    train_accs = []
    test_accs = []
    steps = []

    # Loss function
    def loss_fn(p, x, y):
        y_pred = model(x, p)
        return binary_cross_entropy(y, y_pred), y_pred

    # JIT-compiled update step
    @jax.jit
    def update_step(state, epoch):
        params, opt_state, metrics = state

        (loss_val, y_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_train, y_train)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        train_acc = accuracy(y_train, y_pred)
        test_loss, test_acc = evaluate(model, new_params, x_test, y_test)

        return (new_params, new_opt_state, metrics), (loss_val, train_acc, test_loss, test_acc)

    # Training loop
    state = (params, opt_state, None)
    start = time.time()
    
    for epoch in range(n_epochs):
        state, (train_cost, train_acc, test_cost, test_acc) = update_step(state, epoch)
        
        # Store metrics
        train_costs.append(float(train_cost))
        train_accs.append(float(train_acc))
        test_costs.append(float(test_cost))
        test_accs.append(float(test_acc))
        steps.append(epoch + 1)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {train_cost:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_cost:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    training_time = time.time() - start
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'step': steps,
        'train_cost': train_costs,
        'train_acc': train_accs,
        'test_cost': test_costs,
        'test_acc': test_accs,
        'n_train': [n_train] * len(steps)
    })
    
    return results_df

# Constants
n_test = 100
n_epochs = 100
n_reps = 5
train_sizes = [2, 5, 10, 20, 40, 80]

def run_iterations(n_train):
    """Run multiple training iterations for a given training size and print progress."""
    all_results = []
    for rep in range(n_reps):
        print(f"\nStarting repetition {rep + 1}/{n_reps} for train size {n_train}")
        results_df = train_qvit(n_train, n_test, n_epochs)
        all_results.append(results_df)
    
    return pd.concat(all_results, ignore_index=True)

# Run experiments and collect results
all_results = []
for n_train in train_sizes:
    print(f"\n=== Starting training for train size {n_train} ===")
    results = run_iterations(n_train)
    all_results.append(results)

# Combine all results
results_df = pd.concat(all_results, ignore_index=True)

# Aggregate results
df_agg = results_df.groupby(["n_train", "step"]).agg({
    "train_cost": ["mean", "std"],
    "test_cost": ["mean", "std"],
    "train_acc": ["mean", "std"],
    "test_acc": ["mean", "std"]
}).reset_index()

# Plotting
sns.set_style('whitegrid')
colors = sns.color_palette()
fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))

generalization_errors = []

# Plot losses and accuracies
for i, n_train in enumerate(train_sizes):
    df = df_agg[df_agg.n_train == n_train]
    dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
    lines = ["o-", "x--", "o-", "x--"]
    labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
    axs = [0, 0, 2, 2]

    for k in range(4):
        ax = axes[axs[k]]
        ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=10, color=colors[i], alpha=0.8)

    # Compute generalization error
    dif = df[df.step == n_epochs].test_cost["mean"].values[0] - df[df.step == n_epochs].train_cost["mean"].values[0]
    generalization_errors.append(dif)

# Format plots
axes[0].set_title('Train and Test Losses (CIFAR-10)', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')

axes[1].plot(train_sizes, generalization_errors, "o-", label=r"$gen(\alpha)$")
axes[1].set_xscale('log')
axes[1].set_xticks(train_sizes)
axes[1].set_xticklabels(train_sizes)
axes[1].set_title(r'Generalization Error $gen(\alpha) = R(\alpha) - \hat{R}_N(\alpha)$', fontsize=14)
axes[1].set_xlabel('Training Set Size')
axes[1].set_yscale('log', base=2)

axes[2].set_title('Train and Test Accuracies (CIFAR-10)', fontsize=14)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].set_ylim(0.5, 1.05)

legend_elements = (
    [mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)] +
    [
        mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
        mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
    ]
)

axes[0].legend(handles=legend_elements, ncol=3)
axes[2].legend(handles=legend_elements, ncol=3)

plt.tight_layout()
plt.savefig('qvit_cifar_learning_curves.png')
plt.close()

# Save results to CSV
results_df.to_csv('qvit_cifar_results.csv', index=False)
print("Results saved to qvit_cifar_results.csv")
print("Plots saved to qvit_cifar_learning_curves.png")