#!/usr/bin/env python
# coding: utf-8

# ## Experiment
# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
# from sklearn.datasets import fetch_openml # No longer used directly
# from sklearn.model_selection import train_test_split # No longer used directly
import time
import pandas as pd
# from filelock import FileLock # No longer used directly
import numpy as np
# from jax.experimental import host_callback # No longer used directly
# import tensorflow as tf  # For loading CIFAR-10 # Will be removed
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
    def __init__(self, S, n, Denc, D):
        self.seq_num = S  # Number of sequence positions (16 for 4x4 patches)
        self.num_q = n    # Number of qubits
        self.Denc = Denc  # Depth of encoding ansatz
        self.D = D        # Depth of Q, K, V ansatzes
        self.d = 48       # Dimension of input/output vectors (8x8x3 patches)
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
        # Amplitude encoding with padding or truncation to match 2^num_qubits length
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length:
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        # Normalize input vector, handling zero norm case
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / norm, jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
        # idx = 0
        # for i in range(self.Denc):
        #     for j in range(self.num_q):
        #         qml.CNOT(wires=(j, (j + 1) % self.num_q))
        #     for j in range(self.num_q):
        #         qml.RY(inputs[idx % len(inputs)], wires=j)
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
        return [qml.expval(obs) for obs in self.observables]

    def circuit_qk(self, inputs, weights):
        """Query/Key circuit returning Pauli-Z expectation on qubit 0."""
        # Amplitude encoding with padding or truncation to match 2^num_qubits length
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length:
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        # Normalize input vector, handling zero norm case
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / (norm + 1e-9), jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
        # idx = 0
        # for i in range(self.Denc):
        #     for j in range(self.num_q):
        #         qml.CNOT(wires=(j, (j + 1) % self.num_q))
        #     for j in range(self.num_q):
        #         qml.RY(inputs[idx % len(inputs)], wires=j)
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

    def __call__(self, input_sequence, layer_params):
        # layer_params contains: Q, K, V circuit weights, ln1_gamma, ln1_beta, ffn_w1, ffn_b1, ffn_w2, ffn_b2, ln2_gamma, ln2_beta
        batch_size = input_sequence.shape[0]
        S = self.seq_num
        d = self.d # This is d_patch, should be 48

        # 1. Layer Norm before QSA
        x_norm1 = layer_norm(input_sequence, layer_params['ln1_gamma'], layer_params['ln1_beta'])
        
        # Reshape input for QSA (as before)
        # Assuming x_norm1 is (batch_size, S, d)
        input_flat = jnp.reshape(x_norm1, (-1, d))  # Flatten batch and sequence dimensions together

        # Compute Q, K, V using vectorized operations (Quantum Self-Attention part)
        Q_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['Q']))(input_flat)).T
        K_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['K']))(input_flat)).T
        V_output_flat = jnp.array(jax.vmap(lambda x_patch: self.vqnod(x_patch, layer_params['V']))(input_flat)).T

        # Reshape back to include sequence dimension
        Q_output = Q_output_flat.reshape(batch_size, S, 1)
        K_output = K_output_flat.reshape(batch_size, S, 1)
        V_output = V_output_flat.reshape(batch_size, S, d)

        # Compute Gaussian self-attention coefficients
        Q_expanded = Q_output[:, :, None, :]
        K_expanded = K_output[:, None, :, :]
        alpha = jnp.exp(-(Q_expanded - K_expanded) ** 2)
        Sum_a = jnp.sum(alpha, axis=2, keepdims=True)
        alpha_normalized = alpha / (Sum_a + 1e-9) # Add epsilon for stability

        # Compute weighted sum of values
        V_output_expanded = V_output[:, None, :, :] # V_output is (batch, S, d), need (batch, 1, S, d) for broadcasting with alpha_normalized (batch, S, S, 1)
                                                # So V_output_expanded should be V_output[:, None, :, :] -> (batch_size, 1, S, d_patch)
                                                # alpha_normalized is (batch_size, S, S, 1)
        weighted_V = alpha_normalized * V_output_expanded # (B,S,S,1) * (B,1,S,d) -> (B,S,S,d) via broadcasting
        qsa_out = jnp.sum(weighted_V, axis=2) # Sum over the K dimension (axis=2) -> (B,S,d)

        # 2. First Residual Connection
        x_after_qsa_res = input_sequence + qsa_out

        # 3. Layer Norm before FFN
        x_norm2 = layer_norm(x_after_qsa_res, layer_params['ln2_gamma'], layer_params['ln2_beta'])

        # 4. Feed-Forward Network
        ffn_out = feed_forward(x_norm2, 
                               layer_params['ffn_w1'], layer_params['ffn_b1'], 
                               layer_params['ffn_w2'], layer_params['ffn_b2'])

        # 5. Second Residual Connection
        output = x_after_qsa_res + ffn_out
        return output

class QSANN_pennylane:
    def __init__(self, S, n, Denc, D, num_layers):
        self.qsal_lst = [QSAL_pennylane(S, n, Denc, D) for _ in range(num_layers)]

    def __call__(self, input_sequence, qnn_params_dict):
        # qnn_params_dict contains 'pos_encoding' and 'layers' (list of layer_param dicts)
        x = input_sequence + qnn_params_dict['pos_encoding'] # Add positional encoding
        
        for i, qsal_layer in enumerate(self.qsal_lst):
            layer_specific_params = qnn_params_dict['layers'][i]
            x = qsal_layer(x, layer_specific_params)
        return x

class QSANN_image_classifier:
    def __init__(self, S, n, Denc, D, num_layers):
        self.Qnn = QSANN_pennylane(S, n, Denc, D, num_layers)
        # self.d = n * (Denc + 2) # This d was for the old param count for Q/K/V, not patch dim
        self.d_patch = 48 # Patch dimension
        self.S = S
        self.num_layers = num_layers
        self.final_ln_gamma = None # Will be initialized in init_params for the final LayerNorm
        self.final_ln_beta = None  # Will be initialized in init_params for the final LayerNorm

    def __call__(self, x, params):
        # QNN
        qnn_params_dict = params['qnn'] # This now contains 'pos_encoding' and 'layers'
        x = self.Qnn(x, qnn_params_dict)
        
        # Final Layer Norm before classification head
        x = layer_norm(x, params['final_ln_gamma'], params['final_ln_beta'])
        
        # Flatten
        x = x.reshape(x.shape[0], -1) # (batch_size, S * d_patch)
        # Final layer
        w = params['final']['weight']
        b = params['final']['bias']
        logits = jnp.dot(x, w) + b
        return logits # Return raw logits for cross-entropy

# Loss and Metrics
def softmax_cross_entropy_with_integer_labels(logits, labels):
    """Computes softmax cross entropy between logits and labels (integers)."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels.squeeze())

def accuracy_multiclass(logits, labels):
    """Computes accuracy for multi-class classification."""
    predicted_class = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_class == labels.squeeze()) #.squeeze() to match shape if labels are (N,1)

# Evaluation Function
def evaluate(model, params, dataloader):
    """Evaluate the model on the given dataloader (multi-class)."""
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    cifar_mean_np = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    cifar_std_np = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)

    for x_batch_torch, y_batch_torch in dataloader:
        # Convert PyTorch tensors to NumPy arrays
        x_batch_np = x_batch_torch.numpy()
        y_batch_np = y_batch_torch.numpy().reshape(-1, 1) # Ensure labels are (N, 1)
        current_batch_size = x_batch_np.shape[0]

        # Denormalize: (normalized_image * std) + mean
        x_batch_np = (x_batch_np * cifar_std_np) + cifar_mean_np
        x_batch_np = np.clip(x_batch_np, 0., 1.) # Ensure values are in [0,1]
        x_batch_np = x_batch_np.transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)

        # Create patches
        x_batch_patches_np = create_patches(x_batch_np)

        # Convert to JAX arrays
        x_batch_jax = jnp.array(x_batch_patches_np)
        y_batch_jax = jnp.array(y_batch_np) # Labels are integers 0-9

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

# def augment_image(image): # Removed TensorFlow-based augmentation
#     """Apply random data augmentation to a single image (TensorFlow tensor)."""
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_brightness(image, max_delta=0.1)
#     image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
#     return image

def load_cifar_data(n_train, n_test, batch_size, augment=True):
    """Load and preprocess CIFAR-10 dataset (10 classes) with optional data augmentation.
    Returns PyTorch DataLoaders for training and testing.
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616) # More common std for CIFAR10

    transform_train_list = [transforms.ToTensor()]
    if augment:
        transform_train_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ])
    transform_train_list.append(transforms.Normalize(cifar_mean, cifar_std))
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    # Load CIFAR-10 (10 classes)
    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)
    testset_full = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

    # Subsample if n_train or n_test is smaller than the full dataset
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
        
    # Create DataLoaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0, pin_memory=True) # num_workers=0 for JAX compatibility
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, # Can use a larger batch for testing if memory allows
                                             shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

# Parameter Initialization
def init_params(S, n, Denc, D, num_layers):
    key = jax.random.PRNGKey(42)
    d_patch = 48  # Dimension of input patch vectors
    d_ffn = d_patch * 4  # Inner dimension of FFN

    # Recalculating key splitting based on actual random initializations:
    # Pos enc (1) + num_layers * (Q, K, V, FFN_w1, FFN_w2) (5 per layer) + final head (1 for weight)
    # Final LN params (gamma, beta) are not random. Final bias for head also not random.
    keys = jax.random.split(key, 1 + num_layers * 5 + 1) # +1 for final_weight
    key_idx = 0

    pos_encoding_params = jax.random.normal(keys[key_idx], (S, d_patch), dtype=jnp.float32) * 0.02
    key_idx += 1

    qnn_layers_params = []
    for i in range(num_layers):
        layer_params = {
            'Q': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx], (n * (D + 2),), dtype=jnp.float32) - 1),
            'K': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+1], (n * (D + 2),), dtype=jnp.float32) - 1),
            'V': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+2], (n * (D + 2),), dtype=jnp.float32) - 1),
            'ln1_gamma': jnp.ones((d_patch,), dtype=jnp.float32), # Initialized to 1
            'ln1_beta': jnp.zeros((d_patch,), dtype=jnp.float32),  # Initialized to 0
            'ffn_w1': jax.random.normal(keys[key_idx+3], (d_patch, d_ffn), dtype=jnp.float32) * jnp.sqrt(1.0 / d_patch),
            'ffn_b1': jnp.zeros((d_ffn,), dtype=jnp.float32),
            'ffn_w2': jax.random.normal(keys[key_idx+4], (d_ffn, d_patch), dtype=jnp.float32) * jnp.sqrt(1.0 / d_ffn),
            'ffn_b2': jnp.zeros((d_patch,), dtype=jnp.float32),
            'ln2_gamma': jnp.ones((d_patch,), dtype=jnp.float32), # Initialized to 1
            'ln2_beta': jnp.zeros((d_patch,), dtype=jnp.float32)   # Initialized to 0
        }
        qnn_layers_params.append(layer_params)
        key_idx += 5 # Consumed 5 keys: Q, K, V, ffn_w1, ffn_w2

    params = {
        'qnn': {
            'pos_encoding': pos_encoding_params,
            'layers': qnn_layers_params
        },
        'final_ln_gamma': jnp.ones((d_patch,), dtype=jnp.float32), # Final LN before classifier head
        'final_ln_beta': jnp.zeros((d_patch,), dtype=jnp.float32),
        'final': {
            'weight': jax.random.normal(keys[key_idx], (d_patch * S, 10), dtype=jnp.float32) * 0.01, # Classifier head weights
            'bias': jnp.zeros((10,), dtype=jnp.float32)  # Classifier head bias
        }
    }
    return params


# Training Function
def train_qvit(n_train, n_test, n_epochs, batch_size=64):
    # Load data
    train_loader, test_loader = load_cifar_data(n_train, n_test, batch_size)
    cifar_mean_np = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    cifar_std_np = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)

    # Initialize model and parameters
    model = QSANN_image_classifier(S=64, n=5, Denc=2, D=1, num_layers=1)
    params = init_params(S=64, n=5, Denc=2, D=1, num_layers=1)
    
    # Define optimizer with cosine annealing learning rate schedule
    initial_lr = 0.001
    # Calculate decay_steps based on number of epochs and batches per epoch
    num_batches_per_epoch = len(train_loader) # Approx if last batch is smaller
    decay_steps = n_epochs * num_batches_per_epoch
    lr_schedule = optax.cosine_decay_schedule(init_value=initial_lr, decay_steps=decay_steps)
    optimizer = optax.adam(learning_rate=lr_schedule)

    opt_state = optimizer.init(params)

    # Create arrays to store metrics
    train_costs = []
    test_costs = []
    train_accs = []
    test_accs = []
    steps = []

    # Loss function (batch context)
    def loss_fn_batch(p, x_batch, y_batch):
        logits = model(x_batch, p)
        loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_batch))
        return loss, logits

    # JIT-compiled update step for a single batch
    @jax.jit
    def update_batch(params, opt_state, x_batch_jax, y_batch_jax):
        (loss_val, logits), grads = jax.value_and_grad(loss_fn_batch, has_aux=True)(params, x_batch_jax, y_batch_jax)
        updates, new_opt_state = optimizer.update(grads, opt_state, params) # Pass params for AdamW style updates if needed
        new_params = optax.apply_updates(params, updates)
        batch_acc = accuracy_multiclass(logits, y_batch_jax)
        return new_params, new_opt_state, loss_val, batch_acc

    # Training loop
    current_params = params
    current_opt_state = opt_state
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_train_batches = 0

        for x_batch_torch, y_batch_torch in train_loader:
            # Convert PyTorch tensors to NumPy arrays
            x_batch_np = x_batch_torch.numpy()
            y_batch_np = y_batch_torch.numpy().reshape(-1, 1) # Ensure labels are (N, 1)

            # Denormalize, transpose, and patch
            x_batch_np = (x_batch_np * cifar_std_np) + cifar_mean_np
            x_batch_np = np.clip(x_batch_np, 0., 1.)
            x_batch_np = x_batch_np.transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
            x_batch_patches_np = create_patches(x_batch_np)
            
            # Convert to JAX arrays
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

        # Evaluate on test set at the end of each epoch
        test_loss, test_acc = evaluate(model, current_params, test_loader)
        
        # Store metrics
        train_costs.append(float(avg_epoch_train_loss))
        train_accs.append(float(avg_epoch_train_acc))
        test_costs.append(float(test_loss))
        test_accs.append(float(test_acc))
        steps.append(epoch + 1)
        
        # Print progress (e.g., every epoch or every N epochs)
        if (epoch + 1) % 1 == 0: # Print every epoch for now
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {avg_epoch_train_loss:.4f} | "
                  f"Train Acc: {avg_epoch_train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds for n_train={n_train}")

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'step': steps,
        'train_cost': train_costs,
        'train_acc': train_accs,
        'test_cost': test_costs,
        'test_acc': test_accs,
        'n_train': [n_train] * len(steps),
        'batch_size': [batch_size] * len(steps) # Add batch_size to results
    })
    
    return results_df

# Constants
# n_test = 2000
n_epochs = 100
n_reps = 1 # Consider reducing for faster testing of 10-class setup
train_sizes = [10000] # Consider smaller sizes first for 10-class
BATCH_SIZE = 64 # Define a global batch size or pass it around

def run_iterations(n_train, current_batch_size):
    """Run multiple training iterations for a given training size and print progress."""
    all_results = []
    for rep in range(n_reps):
        print(f"\nStarting repetition {rep + 1}/{n_reps} for train size {n_train}, batch size {current_batch_size}")
        results_df = train_qvit(n_train, n_train//5, n_epochs, batch_size=current_batch_size)
        all_results.append(results_df)
    
    return pd.concat(all_results, ignore_index=True)

# Run experiments and collect results
all_results_collected = [] # Renamed to avoid conflict with all_results in run_iterations
for n_train_current in train_sizes:
    print(f"\n=== Starting training for train size {n_train_current}, batch size {BATCH_SIZE} ===")
    results = run_iterations(n_train_current, BATCH_SIZE)
    all_results_collected.append(results)

# Combine all results
results_df_combined = pd.concat(all_results_collected, ignore_index=True) # Renamed

# Aggregate results
df_agg = results_df_combined.groupby(["n_train", "batch_size", "step"]).agg({ # Added batch_size to groupby
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
results_df_combined.to_csv('qvit_cifar10_results.csv', index=False) # Updated filename
print("Results saved to qvit_cifar10_results.csv")
print("Plots saved to qvit_cifar_learning_curves.png") # Consider updating plot filename if it reflects 10-class