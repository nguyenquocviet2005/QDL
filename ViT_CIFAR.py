#!/usr/bin/env python
# coding: utf-8

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu" # Ensure CPU is used if no GPU preference

import jax
import jax.numpy as jnp
import optax
import time
import pandas as pd
import numpy as np
import tensorflow as tf # For loading CIFAR-10 and data augmentation
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib as mpl

from jax import config
config.update("jax_enable_x64", True)

print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# --- Classical Vision Transformer Components ---

def layer_norm(x, eps=1e-5):
    """Applies Layer Normalization."""
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + eps)

class ClassicalMultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

    def __call__(self, x, params_mhsa):
        """
        x: input sequence (batch_size, seq_len, d_model)
        params_mhsa: dictionary containing 'Wq', 'Wk', 'Wv', 'Wo'
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections for Q, K, V
        q = jnp.einsum('bsd,dhH->bshH', x, params_mhsa['Wq'].reshape(self.d_model, self.num_heads, self.d_head))
        k = jnp.einsum('bsd,dhH->bshH', x, params_mhsa['Wk'].reshape(self.d_model, self.num_heads, self.d_head))
        v = jnp.einsum('bsd,dhH->bshH', x, params_mhsa['Wv'].reshape(self.d_model, self.num_heads, self.d_head))
        
        # Transpose for multi-head attention: (batch_size, num_heads, seq_len, d_head)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.d_head)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attention_output = jnp.matmul(attention_weights, v) # (batch_size, num_heads, seq_len, d_head)

        # Concatenate heads and apply output projection
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3)) # (batch_size, seq_len, num_heads, d_head)
        attention_output = attention_output.reshape(batch_size, seq_len, self.d_model)
        
        output = jnp.dot(attention_output, params_mhsa['Wo'])
        return output

class PositionWiseFFN:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

    def __call__(self, x, params_ffn):
        # params_ffn should contain 'w1', 'b1', 'w2', 'b2'
        h = jnp.dot(x, params_ffn['w1']) + params_ffn['b1']
        h = jax.nn.relu(h)
        output = jnp.dot(h, params_ffn['w2']) + params_ffn['b2']
        return output

class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.mhsa = ClassicalMultiHeadSelfAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)

    def __call__(self, x, params_layer):
        # params_layer should contain 'mhsa' and 'ffn' parameters
        
        # Sublayer 1: Multi-Head Self-Attention (Pre-LN)
        attn_input = layer_norm(x)
        attn_output = self.mhsa(attn_input, params_layer['mhsa'])
        x = x + attn_output # Residual

        # Sublayer 2: Position-wise FFN (Pre-LN)
        ffn_input = layer_norm(x)
        ffn_output = self.ffn(ffn_input, params_layer['ffn'])
        x = x + ffn_output # Residual
        return x

class VisionTransformer:
    def __init__(self, S, d_model, num_layers, num_heads, d_ff, num_classes):
        self.S = S # Sequence length (number of patches)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x_patches, params_vit):
        # x_patches: (batch_size, S, patch_dim) where patch_dim should be d_model
        
        # Add learned positional encoding
        pos_embeddings = params_vit['pos_embedding'] # Shape (S, d_model)
        x = x_patches + pos_embeddings # Broadcasting

        # Pass through Transformer Encoder layers
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, params_vit['encoder_layers'][i])
        
        # Final LayerNorm
        x = layer_norm(x)
        
        # Global Average Pooling (or take [CLS] token if used, here we average patch embeddings)
        # x_pooled = jnp.mean(x, axis=1) # (batch_size, d_model)
        
        # Flatten for final MLP (alternative to pooling if using all patch outputs)
        x_flat = x.reshape(x.shape[0], -1)

        # Final classification head
        logits = jnp.dot(x_flat, params_vit['mlp_head']['weight']) + params_vit['mlp_head']['bias']
        return jax.nn.sigmoid(logits) # Sigmoid for binary classification (CIFAR-10 example uses 2 classes)

# --- Data Loading and Preprocessing (from QViT_CIFAR.py) ---
def create_patches(images, patch_size=4):
    batch_size = images.shape[0]
    img_size = 32 # CIFAR-10 image size
    num_patches_per_dim = img_size // patch_size
    num_patches = num_patches_per_dim * num_patches_per_dim # S = 64 for patch_size=4
    
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            patch = images[:, i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size, :]
            patch = patch.reshape(batch_size, -1) # Flatten patch: 4*4*3 = 48
            patches.append(patch)
    
    patches = jnp.stack(patches, axis=1) # (batch_size, num_patches, patch_dim)
    return patches

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image.numpy() # Convert back to numpy for JAX compatibility if needed later

def load_cifar_data(n_train, n_test, binary=True, augment=True):
    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
    
    if binary: # Using two classes (0: airplane, 1: automobile)
        mask_train = (y_train_full[:, 0] == 0) | (y_train_full[:, 0] == 1)
        mask_test = (y_test_full[:, 0] == 0) | (y_test_full[:, 0] == 1)
        X_train_full = X_train_full[mask_train]
        y_train_full = y_train_full[mask_train]
        X_test_full = X_test_full[mask_test]
        y_test_full = y_test_full[mask_test]
        y_train_full = (y_train_full == 1).astype(jnp.float32) # Ensure JAX float type
        y_test_full = (y_test_full == 1).astype(jnp.float32)

    X_train_full = X_train_full.astype('float32') / 255.0
    X_test_full = X_test_full.astype('float32') / 255.0

    indices_train = np.random.choice(len(X_train_full), n_train, replace=False)
    indices_test = np.random.choice(len(X_test_full), n_test, replace=False)
    X_train = X_train_full[indices_train]
    y_train = y_train_full[indices_train]
    X_test = X_test_full[indices_test]
    y_test = y_test_full[indices_test]

    if augment:
        X_train_augmented = np.array([augment_image(tf.convert_to_tensor(img)) for img in X_train])
        X_train = X_train_augmented

    X_train_patches = create_patches(X_train)
    X_test_patches = create_patches(X_test)
    
    return (
        jnp.array(X_train_patches),
        jnp.array(y_train),
        jnp.array(X_test_patches),
        jnp.array(y_test)
    )

# --- Parameter Initialization ---
def init_params_vit(key, S, d_model, num_layers, num_heads, d_ff, num_classes):
    keys = jax.random.split(key, num_layers * 3 + 2) # For pos_emb, mlp_head, and 3 sets of params per layer (MHSA, FFN)
    
    params = {}
    params['pos_embedding'] = jax.random.normal(keys[0], (S, d_model)) * 0.02

    encoder_layers_params = []
    key_idx = 1
    for _ in range(num_layers):
        layer_p = {}
        
        # MHSA params (Wq, Wk, Wv, Wo)
        # Xavier/Glorot initialization for MHSA matrices
        stddev_mhsa = jnp.sqrt(2.0 / (d_model + d_model)) # For Wq, Wk, Wv
        stddev_wo = jnp.sqrt(2.0 / (d_model + d_model))   # For Wo
        
        layer_p['mhsa'] = {
            'Wq': jax.random.normal(keys[key_idx], (d_model, d_model)) * stddev_mhsa,
            'Wk': jax.random.normal(keys[key_idx+1], (d_model, d_model)) * stddev_mhsa,
            'Wv': jax.random.normal(keys[key_idx+2], (d_model, d_model)) * stddev_mhsa,
            'Wo': jax.random.normal(keys[key_idx+3], (d_model, d_model)) * stddev_wo
        }
        key_idx += 4
        
        # FFN params (w1, b1, w2, b2) - Kaiming/He initialization
        limit_w1 = jnp.sqrt(6. / d_model)
        limit_w2 = jnp.sqrt(6. / d_ff)
        layer_p['ffn'] = {
            'w1': jax.random.uniform(keys[key_idx], (d_model, d_ff), minval=-limit_w1, maxval=limit_w1),
            'b1': jnp.zeros((d_ff,)),
            'w2': jax.random.uniform(keys[key_idx+1], (d_ff, d_model), minval=-limit_w2, maxval=limit_w2),
            'b2': jnp.zeros((d_model,))
        }
        key_idx += 2
        encoder_layers_params.append(layer_p)
        
    params['encoder_layers'] = encoder_layers_params
    
    # MLP Head parameters
    # Output dimension of flattened sequence: S * d_model
    mlp_head_in_dim = S * d_model 
    stddev_mlp = jnp.sqrt(1.0 / mlp_head_in_dim) # Xavier for the final layer
    params['mlp_head'] = {
        'weight': jax.random.normal(keys[key_idx], (mlp_head_in_dim, num_classes)) * stddev_mlp,
        'bias': jnp.zeros((num_classes,))
    }
    
    return params

# --- Loss, Metrics, Evaluation (from QViT_CIFAR.py) ---
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-7
    y_pred = jnp.clip(y_pred, eps, 1.0 - eps)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def accuracy(y_true, y_pred):
    return jnp.mean((y_pred > 0.5) == y_true)

def evaluate(model, params, x, y):
    y_pred = model(x, params)
    loss = binary_cross_entropy(y, y_pred)
    acc = accuracy(y, y_pred)
    return loss, acc

# --- Training Function ---
def train_vit(n_train, n_test, n_epochs, model_config):
    key = jax.random.PRNGKey(42) # Master key for reproducibility
    
    # Load data
    x_train, y_train, x_test, y_test = load_cifar_data(n_train, n_test)

    # Initialize model and parameters
    model = VisionTransformer(
        S=model_config['S'], 
        d_model=model_config['d_model'], 
        num_layers=model_config['num_layers'], 
        num_heads=model_config['num_heads'], 
        d_ff=model_config['d_ff'], 
        num_classes=model_config['num_classes']
    )
    params_key, training_key = jax.random.split(key)
    params = init_params_vit(
        params_key, 
        S=model_config['S'], 
        d_model=model_config['d_model'], 
        num_layers=model_config['num_layers'], 
        num_heads=model_config['num_heads'], 
        d_ff=model_config['d_ff'], 
        num_classes=model_config['num_classes']
    )
    
    initial_lr = model_config['learning_rate']
    lr_schedule = optax.cosine_decay_schedule(init_value=initial_lr, decay_steps=n_epochs * (n_train // model_config.get('batch_size', n_train))) # Decay per step
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=model_config.get('weight_decay', 1e-4))
    opt_state = optimizer.init(params)

    train_costs, test_costs, train_accs, test_accs, steps_log = [], [], [], [], []

    @jax.jit
    def loss_fn_vit(p, x_batch, y_batch):
        y_pred_batch = model(x_batch, p)
        return binary_cross_entropy(y_batch, y_pred_batch), y_pred_batch
    
    @jax.jit
    def update_step(current_params, current_opt_state, x_batch, y_batch):
        (loss_val, y_pred_val), grads_val = jax.value_and_grad(loss_fn_vit, has_aux=True)(current_params, x_batch, y_batch)
        updates, new_opt_state_val = optimizer.update(grads_val, current_opt_state, current_params)
        new_params_val = optax.apply_updates(current_params, updates)
        train_acc_val = accuracy(y_batch, y_pred_val)
        return new_params_val, new_opt_state_val, loss_val, train_acc_val

    batch_size = model_config.get('batch_size', 32 if n_train > 32 else n_train) # Ensure batch_size <= n_train
    num_batches = n_train // batch_size
    
    start_time = time.time()
    current_params_train, current_opt_state_train = params, opt_state

    for epoch in range(n_epochs):
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        
        # Shuffle training data
        perm = jax.random.permutation(training_key, n_train)
        training_key, _ = jax.random.split(training_key) # Update key for next epoch
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            current_params_train, current_opt_state_train, batch_loss, batch_acc = update_step(
                current_params_train, current_opt_state_train, x_batch, y_batch
            )
            epoch_train_loss += batch_loss
            epoch_train_acc += batch_acc
        
        avg_epoch_train_loss = epoch_train_loss / num_batches
        avg_epoch_train_acc = epoch_train_acc / num_batches
        
        test_loss, test_acc = evaluate(model, current_params_train, x_test, y_test)
        
        train_costs.append(float(avg_epoch_train_loss))
        train_accs.append(float(avg_epoch_train_acc))
        test_costs.append(float(test_loss))
        test_accs.append(float(test_acc))
        steps_log.append(epoch + 1)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {avg_epoch_train_loss:.4f} | Train Acc: {avg_epoch_train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    training_time = time.time() - start_time
    print(f"\nTraining completed for n_train={n_train} in {training_time:.2f} seconds")

    return pd.DataFrame({
        'step': steps_log, 'train_cost': train_costs, 'train_acc': train_accs,
        'test_cost': test_costs, 'test_acc': test_accs, 'n_train': [n_train] * len(steps_log)
    })

# --- Main Experiment ---
if __name__ == "__main__":
    # Model Configuration (sensible defaults for CIFAR-10 ViT)
    vit_config = {
        'S': 64,             # Number of patches (32/4 * 32/4)
        'd_model': 48,       # Patch embedding dimension (4*4*3)
        'num_layers': 4,     # Number of Transformer encoder layers
        'num_heads': 6,      # Number of attention heads (d_model % num_heads == 0)
        'd_ff': 4 * 48,      # Hidden dimension of FFN (4 * d_model)
        'num_classes': 1,    # Binary classification (airplane vs automobile)
        'learning_rate': 1e-3,
        'batch_size': 64,
        'weight_decay': 1e-4,
    }

    # Experiment settings
    n_test_samples = 500 # Use a larger test set for more stable evaluation
    num_epochs = 100      # ViTs might need more epochs, but start with 50
    num_repetitions = 5  # Fewer repetitions for faster iteration, can be increased
    training_sizes = [400, 200, 400, 800, 1600, 3200] # Example training sizes

    all_results_list = []
    for n_train_size in training_sizes:
        print(f"\n=== Starting Classical ViT training for train_size = {n_train_size} ===")
        # Ensure batch size is not larger than n_train for this specific run
        current_batch_size = min(vit_config['batch_size'], n_train_size)
        current_vit_config = vit_config.copy()
        current_vit_config['batch_size'] = current_batch_size

        for rep in range(num_repetitions):
            print(f"\nStarting repetition {rep + 1}/{num_repetitions} for train_size {n_train_size}")
            results_df_single_run = train_vit(n_train_size, n_test_samples, num_epochs, current_vit_config)
            results_df_single_run['repetition'] = rep + 1
            all_results_list.append(results_df_single_run)
    
    final_results_df = pd.concat(all_results_list, ignore_index=True)

    # Aggregate results
    df_agg = final_results_df.groupby(["n_train", "step"]).agg(
        train_cost_mean=("train_cost", "mean"), train_cost_std=("train_cost", "std"),
        test_cost_mean=("test_cost", "mean"), test_cost_std=("test_cost", "std"),
        train_acc_mean=("train_acc", "mean"), train_acc_std=("train_acc", "std"),
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std")
    ).reset_index()

    # Plotting (similar to QViT_CIFAR.py)
    sns.set_style('whitegrid')
    palette_colors = sns.color_palette(n_colors=len(training_sizes))
    fig, axes = plt.subplots(ncols=3, figsize=(19.5, 6)) # Increased figure size

    gen_errors_agg = []

    for i, n_train_val_plot in enumerate(training_sizes):
        df_plot = df_agg[df_agg.n_train == n_train_val_plot]
        if df_plot.empty: continue

        axes[0].plot(df_plot.step, df_plot.train_cost_mean, "o-", label=f"N={n_train_val_plot} Train", color=palette_colors[i], markevery=5, alpha=0.8)
        axes[0].plot(df_plot.step, df_plot.test_cost_mean, "x--", label=f"N={n_train_val_plot} Test", color=palette_colors[i], markevery=5, alpha=0.8)
        
        axes[2].plot(df_plot.step, df_plot.train_acc_mean, "o-", label=f"N={n_train_val_plot} Train", color=palette_colors[i], markevery=5, alpha=0.8)
        axes[2].plot(df_plot.step, df_plot.test_acc_mean, "x--", label=f"N={n_train_val_plot} Test", color=palette_colors[i], markevery=5, alpha=0.8)
        
        # Compute generalization error at the end of training for this n_train
        final_epoch_data = df_plot[df_plot.step == num_epochs]
        if not final_epoch_data.empty:
            gen_err = final_epoch_data.test_cost_mean.values[0] - final_epoch_data.train_cost_mean.values[0]
            gen_errors_agg.append(gen_err)
        else:
            gen_errors_agg.append(np.nan)

    axes[0].set_title('ViT: Train and Test Losses (CIFAR-10 Binary)', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(title="Dataset Size (N)", loc="upper right", fontsize='small')

    valid_train_sizes_ge = [ts for i, ts in enumerate(training_sizes) if not np.isnan(gen_errors_agg[i])]
    valid_gen_errors = [ge for ge in gen_errors_agg if not np.isnan(ge)]
    if valid_train_sizes_ge:
        axes[1].plot(valid_train_sizes_ge, valid_gen_errors, "o-", label=r"$gen(\alpha)$")
    axes[1].set_xscale('log')
    axes[1].set_xticks(training_sizes)
    axes[1].set_xticklabels([str(ts) for ts in training_sizes])
    axes[1].set_title(r'ViT: Generalization Error $gen(\alpha)$', fontsize=14)
    axes[1].set_xlabel('Training Set Size (N)')
    axes[1].set_ylabel(r'$R(\alpha) - \hat{R}_N(\alpha)$')
    axes[1].set_yscale('log', base=2, nonpositive='clip') # Clip non-positive for log scale

    axes[2].set_title('ViT: Train and Test Accuracies (CIFAR-10 Binary)', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0.45, 1.05)
    axes[2].legend(title="Dataset Size (N)", loc="lower right", fontsize='small')

    plt.tight_layout(pad=1.5)
    plt.savefig('vit_cifar_learning_curves.png', dpi=300)
    plt.close()

    final_results_df.to_csv('vit_cifar_results.csv', index=False)
    print("\nResults saved to vit_cifar_results.csv")
    print("Plots saved to vit_cifar_learning_curves.png") 