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
    def __init__(self, S, n, Denc, D, num_heads=4):
        self.seq_num = S
        self.num_q = n
        self.Denc = Denc
        self.D = D
        self.d = 192  # Original dimension
        self.num_heads = num_heads
        assert self.d % self.num_heads == 0, "d must be divisible by num_heads"
        self.d_head = self.d // self.num_heads  # Dimension per head

        self.dev = qml.device("default.qubit", wires=self.num_q)

        # Define observables for value circuit (per head)
        # The observables will produce d_head dimensional output from V circuit
        self.observables_head = []
        for i in range(self.d_head):
            qubit = i % self.num_q
            pauli_idx = (i // self.num_q) % 3
            if pauli_idx == 0:
                obs = qml.PauliZ(qubit)
            elif pauli_idx == 1:
                obs = qml.PauliX(qubit)
            else:
                obs = qml.PauliY(qubit)
            self.observables_head.append(obs)

        # Define quantum nodes with JAX interface
        self.vqnod = qml.QNode(self.circuit_v, self.dev, interface="jax")
        self.qnod = qml.QNode(self.circuit_qk, self.dev, interface="jax")

    def circuit_v(self, inputs, weights):
        """Value circuit returning a d_head-dimensional vector of observable expectations."""
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length: # inputs here are features for a single head, d_head
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / norm, jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
        idx = 0
        # Removed Denc loop as per user's latest changes
        # for i in range(self.Denc):
        #     for j in range(self.num_q):
        #         qml.CNOT(wires=(j, (j + 1) % self.num_q))
        #     for j in range(self.num_q):
        #         qml.RY(inputs[idx % len(inputs)], wires=j) # Use inputs for RY based on original paper
        #         idx += 1
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
        return [qml.expval(obs) for obs in self.observables_head]

    def circuit_qk(self, inputs, weights):
        """Query/Key circuit returning Pauli-Z expectation on qubit 0."""
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length: # inputs here are features for a single head, d_head
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)

        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / norm, jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
        idx = 0
        # Removed Denc loop as per user's latest changes
        # for i in range(self.Denc):
        #     for j in range(self.num_q):
        #         qml.CNOT(wires=(j, (j + 1) % self.num_q))
        #     for j in range(self.num_q):
        #         qml.RY(inputs[idx % len(inputs)], wires=j) # Use inputs for RY
        #         idx += 1
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

    def __call__(self, input_sequence, params_heads_Q, params_heads_K, params_heads_V, params_proj):
        batch_size = input_sequence.shape[0]
        S = self.seq_num
        
        # input_sequence shape: (batch_size, S, d)
        # Reshape for multi-head: (batch_size, S, num_heads, d_head)
        # Then permute to (batch_size, num_heads, S, d_head)
        input_reshaped = input_sequence.reshape(batch_size, S, self.num_heads, self.d_head)
        input_permuted = jnp.transpose(input_reshaped, (0, 2, 1, 3)) # (batch_size, num_heads, S, d_head)

        head_outputs = []
        for i in range(self.num_heads):
            # Per-head input: (batch_size, S, d_head)
            head_input = input_permuted[:, i, :, :] 
            # Flatten for vmap: (batch_size * S, d_head)
            head_input_flat = head_input.reshape(-1, self.d_head)

            # Get parameters for the current head
            params_Q_head = params_heads_Q[i]
            params_K_head = params_heads_K[i]
            params_V_head = params_heads_V[i]

            # Compute Q, K, V for the current head
            Q_head_flat = jnp.array(jax.vmap(lambda x: self.qnod(x, params_Q_head))(head_input_flat)).T
            K_head_flat = jnp.array(jax.vmap(lambda x: self.qnod(x, params_K_head))(head_input_flat)).T
            V_head_flat = jnp.array(jax.vmap(lambda x: self.vqnod(x, params_V_head))(head_input_flat)).T
            
            # Reshape back Q, K: (batch_size, S, 1)
            # V: (batch_size, S, d_head)
            Q_head = Q_head_flat.reshape(batch_size, S, 1)
            K_head = K_head_flat.reshape(batch_size, S, 1)
            V_head = V_head_flat.reshape(batch_size, S, self.d_head)

            # Attention for the current head
            alpha_head = jnp.exp(-(Q_head[:, :, None, :] - K_head[:, None, :, :]) ** 2)
            Sum_a_head = jnp.sum(alpha_head, axis=2, keepdims=True)
            alpha_normalized_head = alpha_head / (Sum_a_head + 1e-9) # Add epsilon for stability

            weighted_V_head = alpha_normalized_head * V_head[:, None, :, :]
            Sum_w_head = jnp.sum(weighted_V_head, axis=2) # (batch_size, S, d_head)
            head_outputs.append(Sum_w_head)

        # Concatenate head outputs: (batch_size, S, num_heads * d_head) = (batch_size, S, d)
        concat_heads = jnp.concatenate(head_outputs, axis=-1)

        # Final linear projection: (batch_size, S, d)
        # Reshape for matmul: (batch_size * S, d)
        concat_heads_flat = concat_heads.reshape(-1, self.d)
        projected_flat = jnp.dot(concat_heads_flat, params_proj['weight']) + params_proj['bias']
        projected = projected_flat.reshape(batch_size, S, self.d) # (batch_size, S, d)

        # Add residual connection
        output = input_sequence + projected
        return output

class QSANN_pennylane:
    def __init__(self, S, n, Denc, D, num_layers, num_heads=4):
        self.qsal_lst = [QSAL_pennylane(S, n, Denc, D, num_heads) for _ in range(num_layers)]
        self.num_heads = num_heads

    def __call__(self, input_seq, params):
        x = input_seq
        for i, qsal in enumerate(self.qsal_lst):
            layer_params = params[i]
            x = qsal(x, layer_params['Q_heads'], layer_params['K_heads'], layer_params['V_heads'], layer_params['proj'])
        return x

class QSANN_image_classifier:
    def __init__(self, S, n, Denc, D, num_layers, num_heads=4):
        self.Qnn = QSANN_pennylane(S, n, Denc, D, num_layers, num_heads)
        self.d_model = 48 # This is the d in QSAL_pennylane before head splitting
        self.S = S
        self.num_layers = num_layers
        self.num_heads = num_heads

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

def create_patches(images, patch_size=8):
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
def init_params(S, n, Denc, D, num_layers, num_heads=6):
    key = jax.random.PRNGKey(42)
    d_model = 192 # Original dimension for patches
    d_head = d_model // num_heads

    qnn_params_list = []
    # Each layer has Q, K, V params for each head, and one projection matrix
    num_param_sets_per_layer = num_heads * 3 + 1 
    keys_for_qnn = jax.random.split(key, num_layers * num_param_sets_per_layer + 2) # +2 for final layer and one for splitting key itself
    
    key_idx = 0
    for _ in range(num_layers):
        params_Q_heads = []
        params_K_heads = []
        params_V_heads = []
        for _ in range(num_heads):
            # Parameters for Q, K, V circuits per head
            # Size of weights for QK circuit: n * (D + 2)
            # Size of weights for V circuit: n * (D + 2)
            # Note: Denc is not used for weight count based on current circuit structure
            num_weights_qk = n * (D + 2) 
            num_weights_v = n * (D + 2)

            params_Q_heads.append((jnp.pi / 4) * (2 * jax.random.normal(keys_for_qnn[key_idx], (num_weights_qk,), dtype=jnp.float32) - 1))
            key_idx += 1
            params_K_heads.append((jnp.pi / 4) * (2 * jax.random.normal(keys_for_qnn[key_idx], (num_weights_qk,), dtype=jnp.float32) - 1))
            key_idx += 1
            params_V_heads.append((jnp.pi / 4) * (2 * jax.random.normal(keys_for_qnn[key_idx], (num_weights_v,), dtype=jnp.float32) - 1))
            key_idx += 1
        
        # Parameters for the projection layer (d_model x d_model)
        # Input to projection is concatenated heads (d_model), output is d_model
        proj_weight = 0.01 * jax.random.normal(keys_for_qnn[key_idx], (d_model, d_model), dtype=jnp.float32)
        key_idx +=1
        # Bias for projection layer (d_model) - initialize to zeros
        # For simplicity, often bias is not used in projection if followed by LayerNorm, or can be part of the linear layer.
        # Adding bias for completeness, matching typical linear layers.
        proj_bias = jnp.zeros((d_model,), dtype=jnp.float32)


        qnn_params_list.append({
            'Q_heads': params_Q_heads,
            'K_heads': params_K_heads,
            'V_heads': params_V_heads,
            'proj': {'weight': proj_weight, 'bias': proj_bias}
        })

    final_layer_key_w, final_layer_key_b = keys_for_qnn[key_idx], keys_for_qnn[key_idx+1]
    
    params = {
        'qnn': qnn_params_list,
        'final': {
            'weight': 0.01 * jax.random.normal(final_layer_key_w, (d_model * S, 1), dtype=jnp.float32),
            'bias': jnp.zeros((1,), dtype=jnp.float32) # Or use final_layer_key_b for bias if random
        }
    }
    return params

# Training Function
def train_qvit(n_train, n_test, n_epochs, num_heads=6):
    # Load data
    x_train, y_train, x_test, y_test = load_cifar_data(n_train, n_test)

    # Initialize model and parameters
    model = QSANN_image_classifier(S=16, n=6, Denc=2, D=1, num_layers=1, num_heads=num_heads)
    params = init_params(S=16, n=6, Denc=2, D=1, num_layers=1, num_heads=num_heads)
    
    # Define optimizer with cosine annealing learning rate schedule
    initial_lr = 0.003
    lr_schedule = optax.cosine_decay_schedule(init_value=initial_lr, decay_steps=n_epochs)
    optimizer = optax.adam(learning_rate=lr_schedule)

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
        'n_train': [n_train] * len(steps),
        'num_heads': [num_heads] * len(steps)
    })
    
    return results_df

# Constants
n_test = 100
n_epochs = 100
n_reps = 20
train_sizes = [100,20,40,80,200,400]

def run_iterations(n_train, num_heads=3):
    """Run multiple training iterations for a given training size and print progress."""
    all_results = []
    for rep in range(n_reps):
        print(f"\nStarting repetition {rep + 1}/{n_reps} for train size {n_train}, heads {num_heads}")
        results_df = train_qvit(n_train, n_test, n_epochs, num_heads=num_heads)
        results_df['num_heads'] = num_heads
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
df_agg = results_df.groupby(["n_train", "step", "num_heads"]).agg({
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