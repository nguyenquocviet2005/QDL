#!/usr/bin/env python
# coding: utf-8

# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu" # Ensure CPU execution if no GPU specific setup

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg') # Use non-GUI backend for matplotlib

from jax import config
config.update("jax_enable_x64", True)

# --- Model Configuration Constants (Adapted for CIFAR-10) ---
D_PATCH_VALUE = 96  # From qvt_cifar_nodistill.py
NUM_LAYERS = 1      # From qvt_cifar_nodistill.py
PATCH_SIZE_CIFAR = 4 # CIFAR images are 32x32, 4x4 patches -> 8x8 = 64 patches
S_VALUE_CIFAR = (32 // PATCH_SIZE_CIFAR)**2 # Sequence length (number of patches)
INPUT_PATCH_DIM_CIFAR = (PATCH_SIZE_CIFAR**2) * 3 # For CIFAR-10 (3 channels)
# --- End Model Configuration ---

print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# --- Helper Functions for Transformer Components (from qvt_mnist_nodistill.py) ---
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta

def feed_forward(x, w1, b1, w2, b2):
    x = jnp.dot(x, w1) + b1
    x = jax.nn.relu(x)
    x = jnp.dot(x, w2) + b2
    return x
# --- End Helper Functions ---

# --- QViT Model Classes (from qvt_mnist_nodistill.py, minor adjustments might be needed) ---
class QSAL_pennylane:
    def __init__(self, S, n, Denc, D, d_patch_config):
        self.seq_num = S
        self.num_q = n
        self.Denc = Denc
        self.D = D
        self.d = d_patch_config
        self.dev = qml.device("default.qubit", wires=self.num_q)

        self.observables = []
        for i in range(self.d):
            qubit = i % self.num_q
            pauli_idx = (i // self.num_q) % 3
            if pauli_idx == 0: obs = qml.PauliZ(qubit)
            elif pauli_idx == 1: obs = qml.PauliX(qubit)
            else: obs = qml.PauliY(qubit)
            self.observables.append(obs)

        self.vqnod = qml.QNode(self.circuit_v, self.dev, interface="jax")
        self.qnod = qml.QNode(self.circuit_qk, self.dev, interface="jax")

    def circuit_v(self, inputs, weights):
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length: inputs = inputs[:expected_length]
        elif len(inputs) < expected_length: inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / norm, jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
        idx = 0
        for j in range(self.num_q):
            qml.RX(weights[idx], wires=j); qml.RY(weights[idx + 1], wires=j)
            idx += 2
        for _ in range(self.D):
            for j in range(self.num_q): qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q): qml.RY(weights[idx], wires=j); idx += 1
        return [qml.expval(obs) for obs in self.observables]

    def circuit_qk(self, inputs, weights):
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length: inputs = inputs[:expected_length]
        elif len(inputs) < expected_length: inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / (norm + 1e-9), jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
        idx = 0
        for j in range(self.num_q):
            qml.RX(weights[idx], wires=j); qml.RY(weights[idx + 1], wires=j)
            idx += 2
        for _ in range(self.D):
            for j in range(self.num_q): qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q): qml.RY(weights[idx], wires=j); idx += 1
        return [qml.expval(qml.PauliZ(0))]

    def __call__(self, input_sequence, layer_params):
        batch_size, S, d_in = input_sequence.shape # d_in should be self.d
        x_norm1 = layer_norm(input_sequence, layer_params['ln1_gamma'], layer_params['ln1_beta'])
        input_flat = jnp.reshape(x_norm1, (-1, self.d))

        Q_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['Q']))(input_flat)).T
        K_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['K']))(input_flat)).T
        V_output_flat = jnp.array(jax.vmap(lambda x_patch: self.vqnod(x_patch, layer_params['V']))(input_flat)).T

        Q_output = Q_output_flat.reshape(batch_size, S, 1)
        K_output = K_output_flat.reshape(batch_size, S, 1)
        V_output = V_output_flat.reshape(batch_size, S, self.d)

        alpha = jnp.exp(-(Q_output - K_output) ** 2) # Simplified attention
        Sum_a = jnp.sum(alpha, axis=1, keepdims=True) # Sum over K_output's S dimension
        alpha_normalized = alpha / (Sum_a + 1e-9)
        
        # Weighted sum: alpha is (B, S, 1), V_output is (B, S, d)
        # We need to make alpha (B, S_q, S_kv) and V (B, S_kv, d)
        # Here, S_q = S, S_kv = S. alpha is (B,S,S) effectively if Q,K are distinct sequences
        # In self-attention, Q_output from token i, K_output from token j.
        # Current alpha: (B, S, 1) - K_output used for each Q_output element-wise
        # This implies alpha_ij = exp(-(Q_i - K_j)^2), where Q_i and K_j are scalars.
        # Correct attention:
        # Q_expanded (B, S, 1, 1), K_expanded (B, 1, S, 1) -> alpha (B, S, S, 1)
        # Sum_a over axis=2 (S_kv dimension) -> (B, S, 1, 1)
        # V_output (B, S, d) -> V_output_expanded (B, 1, S, d)
        # weighted_V = alpha_normalized * V_output_expanded -> (B, S, S, d)
        # qsa_out = sum over axis=2 -> (B, S, d)

        Q_exp = Q_output[:, :, None, :] # (B, S_q, 1, 1)
        K_exp = K_output[:, None, :, :] # (B, 1, S_kv, 1)
        alpha_matrix = jnp.exp(-(Q_exp - K_exp)**2) # (B, S_q, S_kv, 1)
        Sum_a_matrix = jnp.sum(alpha_matrix, axis=2, keepdims=True) # (B, S_q, 1, 1)
        alpha_norm_matrix = alpha_matrix / (Sum_a_matrix + 1e-9)

        V_output_exp = V_output[:, None, :, :] # (B, 1, S_kv, d)
        weighted_V = alpha_norm_matrix * V_output_exp # (B, S_q, S_kv, d)
        qsa_out = jnp.sum(weighted_V, axis=2) # (B, S_q, d)

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
        self.d_patch = d_patch_config
        self.S = S
        self.num_layers = num_layers

    def __call__(self, x_patches, params): # x_patches is (B, S, input_patch_dim_actual)
        batch_size, S_actual, input_patch_dim_actual = x_patches.shape
        
        # Linear projection for patches
        x_flat_patches = x_patches.reshape(batch_size * S_actual, input_patch_dim_actual)
        projected_x_flat = jnp.dot(x_flat_patches, params['patch_embed_w']) + params['patch_embed_b']
        x_projected = projected_x_flat.reshape(batch_size, S_actual, self.d_patch)

        qnn_params_dict = params['qnn']
        x_processed_qnn = self.Qnn(x_projected, qnn_params_dict)
        
        x_final_norm = layer_norm(x_processed_qnn, params['final_ln_gamma'], params['final_ln_beta'])
        x_flat_for_head = x_final_norm.reshape(x_final_norm.shape[0], -1)
        
        logits = jnp.dot(x_flat_for_head, params['final']['weight']) + params['final']['bias']
        return logits
# --- End QViT Model Classes ---

# --- Loss and Metrics (from qvt_mnist_nodistill.py) ---
def softmax_cross_entropy_with_integer_labels(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels.squeeze())

def accuracy_multiclass(logits, labels):
    predicted_class = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_class == labels.squeeze())
# --- End Loss and Metrics ---

# --- Patch Creation (Adapted for CIFAR-10) ---
def create_patches(images, patch_size=PATCH_SIZE_CIFAR):
    # images: (B, H, W, C) e.g. (B, 32, 32, 3) for CIFAR
    batch_size = images.shape[0]
    img_size = images.shape[1] # Assuming square images (H=W), 32 for CIFAR
    # num_channels = images.shape[-1] # 3 for CIFAR
    
    num_patches_per_dim = img_size // patch_size
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            patch = images[:, i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size, :] # Include all channels
            patch = patch.reshape(batch_size, -1) # Flatten patch (e.g., 4*4*3 = 48)
            patches.append(patch)
    return jnp.stack(patches, axis=1) # (B, S, patch_dim_flat)
# --- End Patch Creation ---

# --- Data Loading (Adapted for CIFAR-10, similar to qvt_cifar_nodistill.py but for raw 0-1 images) ---
def load_cifar10_data_raw(n_samples, train=True, batch_size=64):
    # No normalization for raw data, attacks expect 0-1 range.
    transform = transforms.Compose([transforms.ToTensor()]) # Converts images to [0,1] range

    dataset_full = torchvision.datasets.CIFAR10(root='./data', train=train,
                                                download=True, transform=transform)
    
    if n_samples < len(dataset_full):
        indices = np.random.choice(len(dataset_full), n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset_full, indices)
    else:
        dataset = dataset_full
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=0) # Shuffle false for consistent test set
    
    # Extract all data from loader
    all_x, all_y = [], []
    for x_batch_torch, y_batch_torch in dataloader:
        all_x.append(x_batch_torch)
        all_y.append(y_batch_torch)
    
    x_all_torch = torch.cat(all_x, dim=0)
    y_all_torch = torch.cat(all_y, dim=0)

    # Convert to numpy, expected format (N, H, W, C) for images
    x_all_np = x_all_torch.numpy().transpose(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)
    y_all_np = y_all_torch.numpy() # (B,)
    
    # Clip to 0-1 just in case, ToTensor should handle this.
    x_all_np = np.clip(x_all_np, 0., 1.)

    return jnp.array(x_all_np), jax.nn.one_hot(jnp.array(y_all_np), 10) # CIFAR-10 has 10 classes

# --- End Data Loading ---

# --- Parameter Initialization (Adapted for CIFAR-10) ---
def init_qvit_params(S, n, Denc, D, num_layers, d_patch_config, input_patch_dim_actual):
    key = jax.random.PRNGKey(42)
    d_ffn = d_patch_config * 4 
    # Num keys: patch_embed_w, patch_embed_b, pos_encoding, Q,K,V (xL), ffn_w1, ffn_b1, ffn_w2, ffn_b2 (xL), final_w, final_b
    # Simplified: 2 (embed) + 1 (pos) + num_layers * (3 QKV + 4 FFN LN) approx - check original
    # From MNIST version: num_random_keys = 2 + 1 + num_layers * 5 + 1 (for final layer head weight only, bias is zeros)
    # Final bias is zeros, final_ln_gamma/beta are ones/zeros, so not random.
    # FFN has w1,b1,w2,b2. MNIST version init only random for w1, w2. b1,b2 are zeros.
    # MNIST: Q,K,V, ffn_w1, ffn_w2 are random per layer. (5 random keys per layer)
    num_random_keys = 2 + 1 + num_layers * 5 + 1 
    keys = jax.random.split(key, num_random_keys)
    key_idx = 0

    # Patch embedding projection from input_patch_dim_actual (e.g. 4*4*3=48) to d_patch_config (e.g. 96)
    patch_embed_w = jax.random.normal(keys[key_idx], (input_patch_dim_actual, d_patch_config), dtype=jnp.float64) * jnp.sqrt(1.0 / input_patch_dim_actual); key_idx+=1
    patch_embed_b = jax.random.normal(keys[key_idx], (d_patch_config,), dtype=jnp.float64) * 0.01; key_idx+=1 # Small random bias
    
    pos_encoding_params = jax.random.normal(keys[key_idx], (S, d_patch_config), dtype=jnp.float64) * 0.02; key_idx+=1

    qnn_layers_params = []
    for _ in range(num_layers):
        layer_params = {
            'Q': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx], (n * (D + 2),), dtype=jnp.float64) - 1),
            'K': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+1], (n * (D + 2),), dtype=jnp.float64) - 1),
            'V': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+2], (n * (D + 2),), dtype=jnp.float64) - 1),
            'ln1_gamma': jnp.ones((d_patch_config,), dtype=jnp.float64), 'ln1_beta': jnp.zeros((d_patch_config,), dtype=jnp.float64),
            'ffn_w1': jax.random.normal(keys[key_idx+3], (d_patch_config, d_ffn), dtype=jnp.float64) * jnp.sqrt(1.0 / d_patch_config), # Kaiming for linear before relu often sqrt(2/fan_in)
            'ffn_b1': jnp.zeros((d_ffn,), dtype=jnp.float64),
            'ffn_w2': jax.random.normal(keys[key_idx+4], (d_ffn, d_patch_config), dtype=jnp.float64) * jnp.sqrt(1.0 / d_ffn),
            'ffn_b2': jnp.zeros((d_patch_config,), dtype=jnp.float64),
            'ln2_gamma': jnp.ones((d_patch_config,), dtype=jnp.float64), 'ln2_beta': jnp.zeros((d_patch_config,), dtype=jnp.float64)
        }
        qnn_layers_params.append(layer_params)
        key_idx += 5

    params = {
        'patch_embed_w': patch_embed_w, 'patch_embed_b': patch_embed_b,
        'qnn': {'pos_encoding': pos_encoding_params, 'layers': qnn_layers_params},
        'final_ln_gamma': jnp.ones((d_patch_config,), dtype=jnp.float64), 
        'final_ln_beta': jnp.zeros((d_patch_config,), dtype=jnp.float64),
        'final': {
            'weight': jax.random.normal(keys[key_idx], (d_patch_config * S, 10), dtype=jnp.float64) * 0.01, # 10 classes for CIFAR-10
            'bias': jnp.zeros((10,), dtype=jnp.float64) # 10 classes for CIFAR-10
        }
    }
    return params
# --- End Parameter Initialization ---

# --- Adversarial Attack Functions (from Adversarial.py, should be mostly dataset agnostic if model_fn is correct) ---
def FGSM(model_fn, params, x, y_one_hot, eps):
  # model_fn(params, x_input) -> logits
  # y_one_hot are labels
  loss_fn_adv = lambda x_in: optax.softmax_cross_entropy(model_fn(params, x_in), y_one_hot).mean()
  grad = jax.grad(loss_fn_adv)(x)
  adv_x = x + eps * jnp.sign(grad)
  return jnp.clip(adv_x, 0.0, 1.0)

def PGD(model_fn, params, x, y_one_hot, eps=8/255, alpha=2/255, steps=5):
  x_adv = x # Use original x, not x.copy() as JAX handles immutability
  loss_fn_adv = lambda x_in: optax.softmax_cross_entropy(model_fn(params, x_in), y_one_hot).mean()
  for _ in range(steps):
      grads = jax.grad(loss_fn_adv)(x_adv)
      x_adv = x_adv + alpha * jnp.sign(grads)
      x_adv = jnp.clip(x_adv, x - eps, x + eps) # Project onto L_inf ball around original x
      x_adv = jnp.clip(x_adv, 0.0, 1.0)      # Project onto valid image range
  return x_adv

def APGD(model_fn, params, x, y_one_hot, eps=0.3, alpha=0.01, steps=10):
  # y_one_hot are one-hot encoded labels, optax.softmax_cross_entropy handles them.
  # loss_fn calculates per-sample losses
  loss_fn = lambda x_in: optax.softmax_cross_entropy(model_fn(params, x_in), y_one_hot)
  # loss_fn_scalar calculates the mean loss for gradient computation
  loss_fn_scalar = lambda x_in: jnp.mean(loss_fn(x_in))

  x_adv = x 
  
  initial_loss = loss_fn(x_adv) # Per-sample losses
  best_loss = initial_loss
  best_adv = x_adv
  # Initialize best_loss_mean_so_far with the mean of the initial per-sample best losses
  best_loss_mean_so_far = jnp.mean(best_loss)

  current_step_size = alpha # Initialize adaptive step size

  for i in range(steps):
      grads = jax.grad(loss_fn_scalar)(x_adv) 
      x_adv = x_adv + current_step_size * jnp.sign(grads) 

      x_adv = jnp.clip(x_adv, x - eps, x + eps)
      x_adv = jnp.clip(x_adv, 0.0, 1.0)

      current_loss_per_sample = loss_fn(x_adv)
      current_loss_mean_batch = jnp.mean(current_loss_per_sample)

      # Adaptive step size rule: if current batch mean loss is not better than 
      # the mean of best losses found so far (per sample), reduce step size.
      # For attacks, "better" means higher loss.
      # The reference code used `cur_loss_mean < best_loss_mean`. 
      # best_loss_mean_so_far is the mean of the maximum losses found for each sample.
      # If current_loss_mean_batch < best_loss_mean_so_far, it implies that on average,
      # this step didn't push losses higher than the established bests.
      if i > 0 and current_loss_mean_batch < best_loss_mean_so_far: # Avoid step reduction on first iteration
          current_step_size = current_step_size * 0.75 # Reduce step size
          # Optional: add a minimum step size if desired
          # current_step_size = jnp.maximum(current_step_size, alpha / 100) 

      update_mask = current_loss_per_sample > best_loss
      if x_adv.ndim > 1 and update_mask.ndim == 1: 
          mask_for_images = update_mask.reshape([-1] + [1] * (x_adv.ndim - 1))
      else:
          mask_for_images = update_mask

      best_adv = jnp.where(mask_for_images, x_adv, best_adv)
      best_loss = jnp.maximum(best_loss, current_loss_per_sample)
      best_loss_mean_so_far = jnp.mean(best_loss) # Update with the mean of new per-sample bests

  return best_adv


def MIM(model_fn, params, x, y_one_hot, eps=8/255, alpha=2/255, steps=10, decay=1.0):
  x_adv = x
  momentum = jnp.zeros_like(x)
  loss_fn_adv = lambda x_in: optax.softmax_cross_entropy(model_fn(params, x_in), y_one_hot).mean()
  for _ in range(steps):
      grad = jax.grad(loss_fn_adv)(x_adv)
      # Normalize grad like in original paper's implementation (often by L1 norm of grad)
      # Original code: grad = grad / (jnp.mean(jnp.abs(grad)) + 1e-8) # This is unusual normalization
      # Standard is often L1 norm for each image's grad:
      # grad_norm = jnp.sum(jnp.abs(grad.reshape(x.shape[0],-1)), axis=1, keepdims=True)
      # grad = grad / (grad_norm.reshape(x.shape[0],1,1,1) + 1e-8)
      # For now, using the user's original MIM grad normalization:
      grad = grad / (jnp.mean(jnp.abs(grad), axis=(1,2,3), keepdims=True) + 1e-12)


      momentum = decay * momentum + grad
      x_adv = x_adv + alpha * jnp.sign(momentum)
      x_adv = jnp.clip(x_adv, x - eps, x + eps)
      x_adv = jnp.clip(x_adv, 0.0, 1.0)
  return x_adv

def SA(model_fn_sa, x, y_one_hot, eps=0.3, num_iters=3, p=0.05, seed=0):
  # model_fn_sa for SA should take x_input -> logits. It's not used by the provided SA code.
  # y_one_hot is also not used by the SA code.
  key = jax.random.PRNGKey(seed)
  adv_x = x 
  
  # SA in Adversarial.py modifies images directly without model feedback.
  # It's a black-box random perturbation attack.
  for i in range(num_iters):
      cur_p = p * (1 - i / num_iters) # Proportion of area to perturb
      # SA in Adversarial.py iterates over batch, JAX typically vmaps or handles batch implicitly.
      # For simplicity, adapting to apply same random patch to all images in batch, or iterate.
      # Let's try to vectorize if possible, or keep loop for clarity matching original.
      # Original loops over batch index `idx`.
      # This is slow if not JITted and run per image.
      # A truly JAX-idiomatic SA would vmap the per-image logic.
      # For now, keeping it simple and potentially slow for large batches:

      # Calculate side_len in the outer loop to make it a concrete value for process_single_image
      # Assuming all images in the batch have the same shape (H, W, C)
      H_batch, W_batch, C_batch = x.shape[1], x.shape[2], x.shape[3]
      area_batch = cur_p * H_batch * W_batch
      # Ensure side_len_concrete is a Python int
      side_len_concrete = int(jnp.maximum(1, jnp.floor(jnp.sqrt(area_batch))).item())


      def process_single_image(img_key_pair, static_side_len, static_eps, static_C_channel):
          img_single, current_key = img_key_pair
          H, W, _ = img_single.shape # C is passed as static_C_channel

          current_key, x0_key, y0_key, pert_key = jax.random.split(current_key, 4)
          
          max_x_offset = H - static_side_len
          max_y_offset = W - static_side_len

          x0 = jax.random.randint(x0_key, (), 0, jnp.maximum(1, max_x_offset + 1))
          y0 = jax.random.randint(y0_key, (), 0, jnp.maximum(1, max_y_offset + 1))
          
          # Use static_side_len and static_C_channel for the shape
          perturbation_shape = (static_side_len, static_side_len, static_C_channel)
          perturbation = jax.random.uniform(pert_key, perturbation_shape, minval=-static_eps, maxval=static_eps)
          
          # Ensure perturbation has the same dtype as img_single before adding
          perturbation = perturbation.astype(img_single.dtype)

          # Get the slice from the original image
          current_slice_val = jax.lax.dynamic_slice(img_single, (x0, y0, 0), perturbation_shape)
          # Add perturbation to the slice
          perturbed_slice_val = current_slice_val + perturbation
          # Update the original image with the perturbed slice
          img_perturbed = jax.lax.dynamic_update_slice(img_single, perturbed_slice_val, (x0, y0, 0))

          return jnp.clip(img_perturbed, 0, 1), current_key
      
      keys_for_batch = jax.random.split(key, x.shape[0] + 1)
      key, img_keys = keys_for_batch[0], keys_for_batch[1:]

      # Use a lambda to pass the concrete side_len for this iteration
      # Also pass eps and C_batch as they are used inside and should be static from this scope
      mapped_fn = lambda ik_pair: process_single_image(ik_pair, side_len_concrete, eps, C_batch)
      adv_x, _ = jax.lax.map(mapped_fn, (adv_x, img_keys))

  return adv_x
# --- End Adversarial Attack Functions ---

# --- Minimal Training and Evaluation (Adapted for CIFAR-10) ---
def get_trained_qvit_params_and_model(n_train_subset=256, n_epochs_subset=3, batch_size_train=64, key_seed=42):
    print(f"Loading CIFAR-10 data for minimal training ({n_train_subset} samples)...")
    # Use the CIFAR-10 data loader for 0-1 range images directly
    x_train_images, y_train_one_hot = load_cifar10_data_raw(n_samples=n_train_subset, train=True, batch_size=n_train_subset)

    print("Initializing QViT model and parameters for CIFAR-10...")
    # Model parameters for CIFAR-10
    # n: num_qubits, Denc: encoding depth, D: ansatz depth
    # For CIFAR-10, S_VALUE_CIFAR and INPUT_PATCH_DIM_CIFAR will be used.
    # D_PATCH_VALUE is the internal dimension of the transformer blocks.
    qvit_model_obj = QSANN_image_classifier(S=S_VALUE_CIFAR, n=5, Denc=2, D=1, # n=5, Denc=2, D=1 are example values
                                          num_layers=NUM_LAYERS, d_patch_config=D_PATCH_VALUE)
    model_params = init_qvit_params(S=S_VALUE_CIFAR, n=5, Denc=2, D=1, num_layers=NUM_LAYERS, 
                                   d_patch_config=D_PATCH_VALUE, input_patch_dim_actual=INPUT_PATCH_DIM_CIFAR)

    initial_lr = 0.001
    optimizer = optax.adam(learning_rate=initial_lr)
    opt_state = optimizer.init(model_params)

    # Re-define with qvit_model_obj from this scope
    @jax.jit
    def update_batch_minimal(params, opt_state, x_batch_patches_jax, y_batch_one_hot_jax):
        # model_obj must be the one initialized in this function scope
        # to ensure it uses CIFAR parameters (S_VALUE_CIFAR etc.)
        def loss_fn(p, x_patches, y_one_hot):
            logits = qvit_model_obj(x_patches, p) # Use qvit_model_obj from outer scope
            y_int_labels = jnp.argmax(y_one_hot, axis=-1)
            loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_int_labels))
            return loss, logits
        
        (loss_val, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_batch_patches_jax, y_batch_one_hot_jax)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    current_params = model_params
    current_opt_state = opt_state
    
    num_batches = int(np.ceil(n_train_subset / batch_size_train))

    print(f"Starting minimal CIFAR-10 training for {n_epochs_subset} epochs...")
    for epoch in range(n_epochs_subset):
        epoch_loss = 0.0
        # Shuffle training data indices for each epoch for simple batching
        perm = np.random.permutation(n_train_subset)
        x_train_images_shuffled = x_train_images[perm]
        y_train_one_hot_shuffled = y_train_one_hot[perm]

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size_train
            end = min((batch_idx + 1) * batch_size_train, n_train_subset)
            
            x_batch_images_np = x_train_images_shuffled[start:end]
            y_batch_one_hot_np = y_train_one_hot_shuffled[start:end]

            # Create patches from (B,H,W,C) CIFAR images
            x_batch_patches_np = create_patches(x_batch_images_np, patch_size=PATCH_SIZE_CIFAR) 
            
            x_batch_patches_jax = jnp.array(x_batch_patches_np)
            y_batch_one_hot_jax = jnp.array(y_batch_one_hot_np)
            
            current_params, current_opt_state, batch_loss = update_batch_minimal(
                current_params, current_opt_state, x_batch_patches_jax, y_batch_one_hot_jax
            )
            epoch_loss += batch_loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{n_epochs_subset}, Avg Loss: {avg_epoch_loss:.4f}")
    print("Minimal CIFAR-10 training finished.")
    return current_params, qvit_model_obj

# evaluate_qvit_on_images needs to use the correct patch_size for CIFAR
def evaluate_qvit_on_images(qvit_model_obj, params, x_images, y_labels_one_hot):
    # x_images: (N, H, W, C) in [0,1] e.g. (N, 32, 32, 3) for CIFAR
    # y_labels_one_hot: (N, num_classes) e.g. (N, 10) for CIFAR
    
    eval_batch_size = 64 
    num_samples = x_images.shape[0]
    num_batches = int(np.ceil(num_samples / eval_batch_size))
    
    total_acc = 0.0
    total_loss = 0.0

    for i in range(num_batches):
        start_idx = i * eval_batch_size
        end_idx = min((i + 1) * eval_batch_size, num_samples)
        
        x_batch_images = x_images[start_idx:end_idx]
        y_batch_one_hot = y_labels_one_hot[start_idx:end_idx]

        # Use CIFAR patch size
        x_batch_patches = create_patches(x_batch_images, patch_size=PATCH_SIZE_CIFAR) 
        
        logits = qvit_model_obj(x_batch_patches, params)
        
        y_batch_int_labels = jnp.argmax(y_batch_one_hot, axis=-1)
        
        acc = accuracy_multiclass(logits, y_batch_int_labels)
        loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_batch_int_labels))
        
        total_acc += acc * x_batch_images.shape[0]
        total_loss += loss * x_batch_images.shape[0]
        
    avg_acc = total_acc / num_samples
    avg_loss = total_loss / num_samples
    return avg_loss, avg_acc
# --- End Minimal Training and Evaluation ---

# --- Main Adversarial Evaluation Script (Adapted for CIFAR-10) ---
if __name__ == "__main__":
    N_TEST_SAMPLES = 200  # Number of test samples for adversarial attacks
    
    print(f"Loading {N_TEST_SAMPLES} CIFAR-10 test samples for adversarial evaluation...")
    x_test_images_jax, y_test_one_hot_jax = load_cifar10_data_raw(n_samples=N_TEST_SAMPLES, train=False, batch_size=N_TEST_SAMPLES)
    print(f"CIFAR-10 Test data loaded: images shape {x_test_images_jax.shape}, labels shape {y_test_one_hot_jax.shape}")

    # Get (minimally) trained QViT model and parameters for CIFAR-10
    # For more robust results, train longer or load pre-trained model.
    # n_train_subset=1024, n_epochs_subset=20 might be slow for CIFAR on CPU without JIT for full training loop.
    # Using smaller values for quick testing. Adjust as needed.
    qvit_params, qvit_model_instance = get_trained_qvit_params_and_model(
        n_train_subset=10000, n_epochs_subset=20, batch_size_train=64
    )

    # Define model function for attacks that use gradients (FGSM, PGD, MIM, APGD)
    # These expect model_fn(params, x_images_batch) -> logits
    # Ensure create_patches inside uses CIFAR patch size
    def qvit_model_fn_for_grad_attacks(params_model, x_images_input):
        x_patches_input = create_patches(x_images_input, patch_size=PATCH_SIZE_CIFAR)
        return qvit_model_instance(x_patches_input, params_model)

    # Define model function for SA attacks
    # SA expects model_fn(x_images_batch) -> logits. Params are bound from the outer scope.
    # Note: SA implementation provided does not actually use this model_fn's output.
    def qvit_model_fn_for_sa_attack(x_images_input):
        x_patches_input = create_patches(x_images_input, patch_size=PATCH_SIZE_CIFAR)
        return qvit_model_instance(x_patches_input, qvit_params) # qvit_params from main scope

    # Evaluate clean accuracy on QViT for CIFAR-10
    print("Evaluating clean accuracy on QViT (CIFAR-10)...")
    _, acc_clean = evaluate_qvit_on_images(qvit_model_instance, qvit_params, x_test_images_jax, y_test_one_hot_jax)
    print(f"Clean accuracy on QViT (CIFAR-10): {acc_clean:.4f}")

    # Configure attacks (eps, alpha, steps might need dataset-specific tuning)
    eps_val = 8/255 # A common starting point for images in [0,1]
    alpha_val = 2/255 # Common PGD step size, often eps/steps_val or similar
    steps_val = 10 # Number of PGD/MIM/APGD steps


    attacks_config = [
        {"name": "FGSM", "func": FGSM, "params": {"eps": eps_val}, "uses_grad_model_fn": True},
        {"name": "PGD", "func": PGD, "params": {"eps": eps_val, "alpha": alpha_val, "steps": steps_val}, "uses_grad_model_fn": True},
        {"name": "MIM", "func": MIM, "params": {"eps": eps_val, "alpha": alpha_val, "steps": steps_val, "decay": 1.0}, "uses_grad_model_fn": True},
        {"name": "APGD", "func": APGD, "params": {"eps": eps_val, "alpha": alpha_val, "steps": steps_val}, "uses_grad_model_fn": True},
        {"name": "SA", "func": SA, "params": {"eps": eps_val, "num_iters": steps_val, "p":0.05, "seed":0}, "uses_grad_model_fn": False}, # SA p might need adjustment for 32x32
    ]

    for attack_item in attacks_config:
        attack_name = attack_item["name"]
        attack_callable = attack_item["func"]
        attack_params_dict = attack_item["params"]
        
        print(f"--- Applying {attack_name} attack on QViT (CIFAR-10) ---")
        
        current_x_test_batch = x_test_images_jax # Using the full test batch loaded
        current_y_test_batch = y_test_one_hot_jax

        if attack_item["uses_grad_model_fn"]:
            x_adv = attack_callable(qvit_model_fn_for_grad_attacks, qvit_params, current_x_test_batch, current_y_test_batch, **attack_params_dict)
        else: # For SA
            # SA's first arg is model_fn, but it's not used by the current SA implementation.
            # Pass y_one_hot as it's in the signature, though also not used.
            x_adv = attack_callable(qvit_model_fn_for_sa_attack, current_x_test_batch, current_y_test_batch, **attack_params_dict)
            
        print(f"Evaluating QViT on {attack_name} adversarial examples (CIFAR-10)...")
        _, acc_adv = evaluate_qvit_on_images(qvit_model_instance, qvit_params, x_adv, current_y_test_batch)
        
        asr = 1.0 - acc_adv  # Attack Success Rate
        robustness_gap = acc_clean - acc_adv

        print(f"Adversarial accuracy ({attack_name}, CIFAR-10): {acc_adv:.4f}")
        print(f"Attack Success Rate ({attack_name}, CIFAR-10): {asr:.4f}")
        print(f"Robustness Gap ({attack_name}, CIFAR-10): {robustness_gap:.4f}")

    print("Adversarial evaluation complete for CIFAR-10.") 