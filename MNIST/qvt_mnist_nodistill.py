#!/usr/bin/env python
# coding: utf-8

# ## Experiment
# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
import time
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-GUI)
import matplotlib.pyplot as plt
import matplotlib as mpl

from jax import config
config.update("jax_enable_x64", True)

# --- Model Configuration Constants ---
D_PATCH_VALUE = 32 # Embedding dimension after patch projection
NUM_LAYERS = 1     # Number of QNN layers
S_VALUE_MNIST = 49 # Number of patches for MNIST (7x7 grid from 28x28 image with 4x4 patches)
INPUT_PATCH_DIM_MNIST = 16 # Dimension of each patch for MNIST (4*4*1)
# --- End Model Configuration ---

# Check JAX backend (e.g., CPU or GPU)
print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# --- Helper Functions for Transformer Components ---
def layer_norm(x, gamma, beta, eps=1e-5):
    """Applies Layer Normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta

def feed_forward(x, w1, b1, w2, b2):
    """Position-wise Feed-Forward Network with ReLU."""
    x = jnp.dot(x, w1) + b1
    x = jax.nn.relu(x)
    x = jnp.dot(x, w2) + b2
    return x
# --- End Helper Functions ---

# QViT Model Classes (Adapted for JAX)
class QSAL_pennylane:
    def __init__(self, S, n, Denc, D, d_patch_config):
        self.seq_num = S  # Number of sequence positions
        self.num_q = n    # Number of qubits
        self.Denc = Denc  # Depth of encoding ansatz
        self.D = D        # Depth of Q, K, V ansatzes
        self.d = d_patch_config # Dimension of input/output vectors (embedding dimension)
        self.dev = qml.device("default.qubit", wires=self.num_q)

        # Define observables for value circuit, now based on d_patch_config
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
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length:
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / norm, jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
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
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length:
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / (norm + 1e-9), jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
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

    def __call__(self, input_sequence, layer_params):
        batch_size = input_sequence.shape[0]
        S = self.seq_num 
        d = self.d # This is d_patch_config (embedding dimension)

        x_norm1 = layer_norm(input_sequence, layer_params['ln1_gamma'], layer_params['ln1_beta'])
        input_flat = jnp.reshape(x_norm1, (-1, d))

        Q_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['Q']))(input_flat)).T
        K_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['K']))(input_flat)).T
        V_output_flat = jnp.array(jax.vmap(lambda x_patch: self.vqnod(x_patch, layer_params['V']))(input_flat)).T

        Q_output = Q_output_flat.reshape(batch_size, S, 1)
        K_output = K_output_flat.reshape(batch_size, S, 1)
        V_output = V_output_flat.reshape(batch_size, S, d)

        Q_expanded = Q_output[:, :, None, :]
        K_expanded = K_output[:, None, :, :]
        alpha = jnp.exp(-(Q_expanded - K_expanded) ** 2)
        Sum_a = jnp.sum(alpha, axis=2, keepdims=True)
        alpha_normalized = alpha / (Sum_a + 1e-9)

        V_output_expanded = V_output[:, None, :, :]
        weighted_V = alpha_normalized * V_output_expanded
        qsa_out = jnp.sum(weighted_V, axis=2)

        x_after_qsa_res = input_sequence + qsa_out
        x_norm2 = layer_norm(x_after_qsa_res, layer_params['ln2_gamma'], layer_params['ln2_beta'])
        ffn_out = feed_forward(x_norm2, 
                               layer_params['ffn_w1'], layer_params['ffn_b1'], 
                               layer_params['ffn_w2'], layer_params['ffn_b2'])
        output = x_after_qsa_res + ffn_out
        return output

class QSANN_pennylane:
    def __init__(self, S, n, Denc, D, num_layers, d_patch_config):
        self.qsal_lst = [QSAL_pennylane(S, n, Denc, D, d_patch_config) for _ in range(num_layers)]

    def __call__(self, input_sequence, qnn_params_dict):
        x = input_sequence + qnn_params_dict['pos_encoding']
        for i, qsal_layer in enumerate(self.qsal_lst):
            layer_specific_params = qnn_params_dict['layers'][i]
            x = qsal_layer(x, layer_specific_params)
        return x

class QSANN_image_classifier:
    def __init__(self, S, n, Denc, D, num_layers, d_patch_config):
        self.Qnn = QSANN_pennylane(S, n, Denc, D, num_layers, d_patch_config)
        self.d_patch = d_patch_config # Store the configured patch dimension (embedding dim)
        self.S = S # Number of patches
        self.num_layers = num_layers

    def __call__(self, x, params):
        # x is initially (batch_size, S, input_patch_dim_actual) e.g. (B, 49, 16) for MNIST
        batch_size, S_actual, input_patch_dim_actual = x.shape
        x_flat = x.reshape(batch_size * S_actual, input_patch_dim_actual)
        projected_x_flat = jnp.dot(x_flat, params['patch_embed_w']) + params['patch_embed_b']
        # x_projected is (batch_size, S, self.d_patch) where self.d_patch is d_patch_config (embedding dim)
        x_projected = projected_x_flat.reshape(batch_size, S_actual, self.d_patch)

        qnn_params_dict = params['qnn']
        x_processed_qnn = self.Qnn(x_projected, qnn_params_dict)
        
        x_final_norm = layer_norm(x_processed_qnn, params['final_ln_gamma'], params['final_ln_beta'])
        x_flat_for_head = x_final_norm.reshape(x_final_norm.shape[0], -1) # (batch_size, S * d_patch)
        
        w = params['final']['weight']
        b = params['final']['bias']
        logits = jnp.dot(x_flat_for_head, w) + b
        return logits

# Loss and Metrics
def softmax_cross_entropy_with_integer_labels(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels.squeeze())

def accuracy_multiclass(logits, labels):
    predicted_class = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_class == labels.squeeze())

# Evaluation Function
def evaluate(model, params, dataloader):
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    mnist_mean_np = np.array([0.1307]).reshape(1, 1, 1, 1) # For (B,C,H,W)
    mnist_std_np = np.array([0.3081]).reshape(1, 1, 1, 1)  # For (B,C,H,W)

    for x_batch_torch, y_batch_torch in dataloader:
        x_batch_np = x_batch_torch.numpy()
        y_batch_np = y_batch_torch.numpy().reshape(-1, 1)
        current_batch_size = x_batch_np.shape[0]

        # Denormalize: (normalized_image * std) + mean
        x_batch_np = (x_batch_np * mnist_std_np) + mnist_mean_np # MNIST is 1 channel
        x_batch_np = np.clip(x_batch_np, 0., 1.)
        x_batch_np = x_batch_np.transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C) -> (B,28,28,1) for MNIST

        x_batch_patches_np = create_patches(x_batch_np) # MNIST patches

        x_batch_jax = jnp.array(x_batch_patches_np)
        y_batch_jax = jnp.array(y_batch_np)

        logits = model(x_batch_jax, params)
        loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_batch_jax))
        acc = accuracy_multiclass(logits, y_batch_jax)

        total_loss += loss * current_batch_size
        total_acc += acc * current_batch_size
        num_samples += current_batch_size

    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    return avg_loss, avg_acc

def create_patches(images, patch_size=4):
    """Convert MNIST images into patches.
    
    Args:
        images: Array of shape (batch_size, 28, 28, 1)
        patch_size: Size of each square patch (e.g., 4)
    
    Returns:
        patches: Array of shape (batch_size, num_patches, patch_size*patch_size*1)
                 e.g. (batch_size, 49, 16) for 28x28 images, 4x4 patches
    """
    batch_size = images.shape[0]
    img_size = 28 # MNIST image size
    num_channels = images.shape[-1] # Should be 1 for MNIST
    
    num_patches_per_dim = img_size // patch_size
    # num_patches = num_patches_per_dim * num_patches_per_dim # This is S_VALUE_MNIST (49)
    
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            patch = images[:, i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size, :]
            patch = patch.reshape(batch_size, -1) # Flatten to (batch_size, patch_size*patch_size*num_channels)
            patches.append(patch)
    
    patches = jnp.stack(patches, axis=1)
    return patches

def load_mnist_data(n_train, n_test, batch_size, augment=True):
    """Load and preprocess MNIST dataset with optional data augmentation.
    Returns PyTorch DataLoaders for training and testing.
    """
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    transform_train_list = [transforms.ToTensor()]
    if augment:
        transform_train_list.extend([
            transforms.RandomCrop(28, padding=2), # Adjusted padding for 28x28
            transforms.RandomHorizontalFlip(),
            # ColorJitter removed as MNIST is grayscale
        ])
    transform_train_list.append(transforms.Normalize(mnist_mean, mnist_std))
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])

    trainset_full = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform_train)
    testset_full = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=transform_test)

    if n_train < len(trainset_full):
        train_indices = np.random.choice(len(trainset_full), n_train, replace=False)
        trainset = torch.utils.data.Subset(trainset_full, train_indices)
    else:
        trainset = trainset_full
    
    if n_test < len(testset_full):
        test_indices = np.random.choice(len(testset_full), n_test, replace=False)
        testset = torch.utils.data.Subset(testset_full, test_indices)
    else:
        testset = testset_full
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

# Parameter Initialization
def init_params(S, n, Denc, D, num_layers, d_patch_config, input_patch_dim):
    key = jax.random.PRNGKey(42)
    # S: number of patches (e.g., 49 for MNIST)
    # input_patch_dim: dimension of each patch (e.g., 16 for MNIST 4x4x1)
    # d_patch_config: target embedding dimension (e.g., D_PATCH_VALUE = 64)
    d_ffn = d_patch_config * 4

    num_random_keys = 2 + 1 + num_layers * 5 + 1
    keys = jax.random.split(key, num_random_keys)
    key_idx = 0

    patch_embed_w = jax.random.normal(keys[key_idx], (input_patch_dim, d_patch_config), dtype=jnp.float32) * jnp.sqrt(1.0 / input_patch_dim)
    key_idx += 1
    patch_embed_b = jax.random.normal(keys[key_idx], (d_patch_config,), dtype=jnp.float32) * 0.01
    key_idx += 1

    pos_encoding_params = jax.random.normal(keys[key_idx], (S, d_patch_config), dtype=jnp.float32) * 0.02
    key_idx += 1

    qnn_layers_params = []
    for i in range(num_layers):
        layer_params = {
            'Q': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx], (n * (D + 2),), dtype=jnp.float32) - 1),
            'K': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+1], (n * (D + 2),), dtype=jnp.float32) - 1),
            'V': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+2], (n * (D + 2),), dtype=jnp.float32) - 1),
            'ln1_gamma': jnp.ones((d_patch_config,), dtype=jnp.float32),
            'ln1_beta': jnp.zeros((d_patch_config,), dtype=jnp.float32),
            'ffn_w1': jax.random.normal(keys[key_idx+3], (d_patch_config, d_ffn), dtype=jnp.float32) * jnp.sqrt(1.0 / d_patch_config),
            'ffn_b1': jnp.zeros((d_ffn,), dtype=jnp.float32),
            'ffn_w2': jax.random.normal(keys[key_idx+4], (d_ffn, d_patch_config), dtype=jnp.float32) * jnp.sqrt(1.0 / d_ffn),
            'ffn_b2': jnp.zeros((d_patch_config,), dtype=jnp.float32),
            'ln2_gamma': jnp.ones((d_patch_config,), dtype=jnp.float32),
            'ln2_beta': jnp.zeros((d_patch_config,), dtype=jnp.float32)
        }
        qnn_layers_params.append(layer_params)
        key_idx += 5

    params = {
        'patch_embed_w': patch_embed_w,
        'patch_embed_b': patch_embed_b,
        'qnn': {
            'pos_encoding': pos_encoding_params,
            'layers': qnn_layers_params
        },
        'final_ln_gamma': jnp.ones((d_patch_config,), dtype=jnp.float32),
        'final_ln_beta': jnp.zeros((d_patch_config,), dtype=jnp.float32),
        'final': { # Classifier head for 10 classes (MNIST)
            'weight': jax.random.normal(keys[key_idx], (d_patch_config * S, 10), dtype=jnp.float32) * 0.01,
            'bias': jnp.zeros((10,), dtype=jnp.float32)
        }
    }
    return params

def count_parameters(params):
    count = 0
    for leaf in jax.tree_util.tree_leaves(params):
        count += leaf.size
    return count

# Training Function
def train_qvit(n_train, n_test, n_epochs, batch_size=64):
    train_loader, test_loader = load_mnist_data(n_train, n_test, batch_size)
    # MNIST specific mean and std for denormalization during training batch processing
    mnist_mean_np_train = np.array([0.1307]).reshape(1, 1, 1, 1) # For (B,C,H,W)
    mnist_std_np_train = np.array([0.3081]).reshape(1, 1, 1, 1)  # For (B,C,H,W)

    model = QSANN_image_classifier(S=S_VALUE_MNIST, n=5, Denc=2, D=1, num_layers=NUM_LAYERS, d_patch_config=D_PATCH_VALUE)
    params = init_params(S=S_VALUE_MNIST, n=5, Denc=2, D=1, num_layers=NUM_LAYERS, d_patch_config=D_PATCH_VALUE, input_patch_dim=INPUT_PATCH_DIM_MNIST)
    
    total_params = count_parameters(params)
    print(f"Total number of parameters in the QViT model for MNIST: {total_params:,}")

    initial_lr = 0.0005
    num_batches_per_epoch = len(train_loader)
    decay_steps = n_epochs * num_batches_per_epoch
    lr_schedule = optax.cosine_decay_schedule(init_value=initial_lr, decay_steps=decay_steps)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    train_costs, test_costs, train_accs, test_accs, steps = [], [], [], [], []

    def loss_fn_batch(p, x_batch, y_batch):
        logits = model(x_batch, p)
        loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_batch))
        return loss, logits

    @jax.jit
    def update_batch(params, opt_state, x_batch_jax, y_batch_jax):
        (loss_val, logits), grads = jax.value_and_grad(loss_fn_batch, has_aux=True)(params, x_batch_jax, y_batch_jax)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        batch_acc = accuracy_multiclass(logits, y_batch_jax)
        return new_params, new_opt_state, loss_val, batch_acc

    current_params = params
    current_opt_state = opt_state
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_train_batches = 0

        for x_batch_torch, y_batch_torch in train_loader:
            x_batch_np = x_batch_torch.numpy()
            y_batch_np = y_batch_torch.numpy().reshape(-1, 1)

            x_batch_np = (x_batch_np * mnist_std_np_train) + mnist_mean_np_train
            x_batch_np = np.clip(x_batch_np, 0., 1.)
            x_batch_np = x_batch_np.transpose(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C) -> (B,28,28,1)
            x_batch_patches_np = create_patches(x_batch_np) # MNIST patches
            
            x_batch_jax = jnp.array(x_batch_patches_np)
            y_batch_jax = jnp.array(y_batch_np)
            current_params, current_opt_state, batch_loss, batch_acc = update_batch(
                current_params, current_opt_state, x_batch_jax, y_batch_jax
            )
            epoch_train_loss += batch_loss
            epoch_train_acc += batch_acc
            num_train_batches += 1
        
        avg_epoch_train_loss = epoch_train_loss / num_train_batches
        avg_epoch_train_acc = epoch_train_acc / num_train_batches

        test_loss, test_acc = evaluate(model, current_params, test_loader)
        
        train_costs.append(float(avg_epoch_train_loss))
        train_accs.append(float(avg_epoch_train_acc))
        test_costs.append(float(test_loss))
        test_accs.append(float(test_acc))
        steps.append(epoch + 1)
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {avg_epoch_train_loss:.4f} | "
                  f"Train Acc: {avg_epoch_train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds for n_train={n_train} (MNIST)")

    results_df = pd.DataFrame({
        'step': steps,
        'train_cost': train_costs,
        'train_acc': train_accs,
        'test_cost': test_costs,
        'test_acc': test_accs,
        'n_train': [n_train] * len(steps),
        'batch_size': [batch_size] * len(steps)
    })
    return results_df

# Constants for MNIST experiment
# MNIST has 60,000 training images, 10,000 test images
# n_test_mnist = 10000 # Using full test set for MNIST
n_epochs_mnist = 100  # MNIST might train faster, adjust as needed
n_reps_mnist = 1
train_sizes_mnist = [1000, 2000, 5000, 10000, 20000] # Example sizes for MNIST
BATCH_SIZE_MNIST = 64

def run_iterations(n_train, current_batch_size, num_epochs_current):
    all_results = []
    for rep in range(n_reps_mnist):
        print(f"\nStarting repetition {rep + 1}/{n_reps_mnist} for train size {n_train}, batch size {current_batch_size} (MNIST)")
        # Ensure n_test is an integer for np.random.choice
        n_test_int = int(n_train // 5) 
        results_df = train_qvit(n_train, n_test_int, num_epochs_current, batch_size=current_batch_size)
        all_results.append(results_df)
    return pd.concat(all_results, ignore_index=True)

all_results_collected_mnist = []
for n_train_current in train_sizes_mnist:
    print(f"=== Starting training for MNIST: train size {n_train_current}, batch size {BATCH_SIZE_MNIST}, epochs {n_epochs_mnist} ===")
    results = run_iterations(n_train_current, BATCH_SIZE_MNIST, n_epochs_mnist)
    all_results_collected_mnist.append(results)

results_df_combined_mnist = pd.concat(all_results_collected_mnist, ignore_index=True)

df_agg_mnist = results_df_combined_mnist.groupby(["n_train", "batch_size", "step"]).agg({
    "train_cost": ["mean", "std"],
    "test_cost": ["mean", "std"],
    "train_acc": ["mean", "std"],
    "test_acc": ["mean", "std"]
}).reset_index()

# Plotting
sns.set_style('whitegrid')
colors = sns.color_palette("viridis", n_colors=len(train_sizes_mnist)) # Different palette for clarity
fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))
generalization_errors_mnist = []

for i, n_train in enumerate(train_sizes_mnist):
    df = df_agg_mnist[(df_agg_mnist.n_train == n_train) & (df_agg_mnist.batch_size == BATCH_SIZE_MNIST)]
    dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
    lines = ["o-", "x--", "o-", "x--"]
    labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
    axs = [0, 0, 2, 2]

    for k in range(4):
        ax = axes[axs[k]]
        ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=max(1, n_epochs_mnist // 10), color=colors[i], alpha=0.8)

    # Compute generalization error at the end of training
    if not df[df.step == n_epochs_mnist].empty:
      train_cost_final = df[df.step == n_epochs_mnist].train_cost["mean"].values[0]
      test_cost_final = df[df.step == n_epochs_mnist].test_cost["mean"].values[0]
      dif = test_cost_final - train_cost_final
      generalization_errors_mnist.append(dif)
    else: # Fallback if exact last epoch not found (should not happen with reset_index)
      generalization_errors_mnist.append(np.nan)


axes[0].set_title('Train and Test Losses (MNIST)', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')

axes[1].plot(train_sizes_mnist, generalization_errors_mnist, "o-", label=r"$gen(\alpha)$")
axes[1].set_xscale('log')
axes[1].set_xticks(train_sizes_mnist)
axes[1].set_xticklabels([str(ts) for ts in train_sizes_mnist]) # Ensure labels are strings
axes[1].set_title(r'Generalization Error $gen(\alpha)$ (MNIST)', fontsize=14)
axes[1].set_xlabel('Training Set Size')
axes[1].set_yscale('log', base=2, nonpositive='clip') # clip nonpositive for log scale

axes[2].set_title('Train and Test Accuracies (MNIST)', fontsize=14)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].set_ylim(0.7, 1.05) # MNIST accuracies are typically higher

legend_elements_mnist = (
    [mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes_mnist)] +
    [
        mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
        mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
    ]
)

axes[0].legend(handles=legend_elements_mnist, ncol=2) # Adjusted ncol for potentially fewer train sizes
axes[2].legend(handles=legend_elements_mnist, ncol=2)

plt.tight_layout()
plt.savefig('qvit_mnist_learning_curves.png')
plt.close()

results_df_combined_mnist.to_csv('qvit_mnist_results.csv', index=False)
print("Results saved to qvit_mnist_results.csv")
print("Plots saved to qvit_mnist_learning_curves.png")

print("MNIST QViT script setup complete.") 