#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import jax
import jax.numpy as jnp
import optax
from tensorflow.keras.datasets import mnist
import tensorflow as tf


# In[67]:


def load_mnist_binary(num_train=200, num_test=200, rng=np.random.RandomState(0)):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    idx = np.where((y == 0) | (y == 1))[0]
    x, y = x[idx], y[idx]

    x = x.astype(np.float32) / 255.0
    x = np.expand_dims(x, -1)

    y = (y == 1).astype(np.int32)  # Make it 0 or 1

    indices = rng.choice(len(x), num_train + num_test, replace=False)
    x, y = x[indices], y[indices]

    x_train, x_test = x[:num_train], x[num_train:]
    y_train, y_test = y[:num_train], y[num_train:]

    return jnp.array(x_train), jax.nn.one_hot(y_train, 2), jnp.array(x_test), jax.nn.one_hot(y_test, 2)



# In[68]:


def init_mlp_params(rng_key):
    keys = jax.random.split(rng_key, 4)
    params = {
        "w1": jax.random.normal(keys[0], (28 * 28, 128)) * 0.01,
        "b1": jnp.zeros(128),
        "w2": jax.random.normal(keys[1], (128, 2)) * 0.01,
        "b2": jnp.zeros(2)
    }
    return params

def model_fn(params, x):
    x_flat = x.reshape((x.shape[0], -1))
    hidden = jax.nn.relu(jnp.dot(x_flat, params["w1"]) + params["b1"])
    logits = jnp.dot(hidden, params["w2"]) + params["b2"]
    return logits


# In[69]:


def FGSM(model_fn, params, x, y, eps):
  loss_fn = lambda x_in: optax.softmax_cross_entropy(model_fn(params, x_in), y).mean()
  grad = jax.grad(loss_fn)(x)
  adv_x = x + eps * jnp.sign(grad)
  return jnp.clip(adv_x, 0.0, 1.0)



def PGD(model_fn, params, x, y, eps=8/255, alpha=2/255, steps=5):
  x_adv = x.copy()
  loss_fn = lambda x_in: optax.softmax_cross_entropy(model_fn(params, x_in), y).mean()
  for _ in range(steps):
      grads = jax.grad(loss_fn)(x_adv)
      x_adv = x_adv + alpha * jnp.sign(grads)
      x_adv = jnp.clip(x_adv, x - eps, x + eps)
      x_adv = jnp.clip(x_adv, 0.0, 1.0)
  return x_adv

def APGD(model_fn, params, x, y, eps=0.3, alpha=0.01, steps=10, seed=0):
  loss_fn = lambda x_in: optax.softmax_cross_entropy(model_fn(params, x_in), y)
  loss_fn_scalar =  lambda x_in: optax.softmax_cross_entropy(model_fn(params, x_in), y).mean()
  x_adv = x.copy()
  step_size = alpha
  cur_loss = loss_fn(x_adv)
  cur_loss_mean = cur_loss.mean()
  best_loss = cur_loss
  best_loss_mean = cur_loss_mean
  best_adv = x_adv

  for _ in range(steps):
    grads = jax.grad(loss_fn_scalar)(x_adv)
    x_adv = x_adv + alpha * jnp.sign(grads)

    x_adv = jnp.clip(x_adv, x - eps, x + eps)
    x_adv = jnp.clip(x_adv, 0.0, 1.0)

    cur_loss = loss_fn(x_adv)
    cur_loss_mean = cur_loss.mean()
    if cur_loss_mean < best_loss_mean:
        step_size = step_size * 0.75

    update_mask = cur_loss > best_loss
    update_mask = update_mask[:, None, None, None]

    best_adv = jnp.where(update_mask, x_adv, best_adv)
    best_loss = jnp.maximum(best_loss, cur_loss)
    best_loss_mean = best_loss.mean()
  return best_adv

def MIM(model_fn, params, x, y, eps=8/255, alpha=2/255, steps=10, decay=1.0):
  x_adv = x.copy()
  momentum = jnp.zeros_like(x)
  loss_fn = lambda x_in:optax.softmax_cross_entropy(model_fn(params, x_in), y).mean()
  for _ in range(steps):

      grad = jax.grad(loss_fn)(x_adv)
      grad = grad / (jnp.mean(jnp.abs(grad)) + 1e-8)

      momentum = decay * momentum + grad
      x_adv = x_adv + alpha * jnp.sign(momentum)
      x_adv = jnp.clip(x_adv, x - eps, x + eps)
  return jnp.clip(x_adv, 0, 1)
def SA(model_fn, x, y, eps=0.3, num_iters=3, p=0.05, seed=0):
  key = jax.random.PRNGKey(seed)
  adv_x = x.copy()
  for i in range(num_iters):
      cur_p = p * (1 - i / num_iters)
      for idx in range(x.shape[0]):
          img = adv_x[idx]
          H, W, C = img.shape
          area = int(cur_p * H * W)
          side_len = max(1, int(jnp.sqrt(area)))

          key, rand = jax.random.split(key)
          x0 = jax.random.randint(rand, (), 0, H - side_len)
          y0 = jax.random.randint(rand, (), 0, W - side_len)

          key, rand = jax.random.split(key)
          perturbation = jax.random.uniform(rand, (side_len, side_len, C), minval=-eps, maxval=eps)

          img = img.at[x0:x0+side_len, y0:y0+side_len, :].add(perturbation)
          img = jnp.clip(img, 0, 1)
          adv_x = adv_x.at[idx].set(img)
  return adv_x


# In[70]:


def train_model(x_train, y_train, num_epochs=10, lr=0.1):
    params = init_mlp_params(jax.random.PRNGKey(0))
    optimizer = optax.sgd(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, x, y):
        logits = model_fn(params, x)
        return optax.softmax_cross_entropy(logits, y).mean()

    @jax.jit
    def update(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    for epoch in range(num_epochs):
        params, opt_state, loss = update(params, opt_state, x_train, y_train)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    return params

def evaluate(params, x, y):
    logits = model_fn(params, x)
    preds = jnp.argmax(logits, axis=1)
    labels = jnp.argmax(y, axis=1)
    return jnp.mean(preds == labels)


# In[71]:


x_train, y_train, x_test, y_test = load_mnist_binary()
params = train_model(x_train, y_train)


# In[72]:


print ("FGSM attempt")
acc_clean = evaluate(params, x_test, y_test)
print(f"Clean accuracy: {acc_clean:.4f}")

x_adv = FGSM(model_fn, params, x_test, y_test, eps=0.2)
acc_adv = evaluate(params, x_adv, y_test)

asr = 1.0 - acc_adv
robustness_gap = acc_clean - acc_adv

print(f"Adversarial accuracy: {acc_adv:.4f}")
print(f"Attack Success Rate: {asr:.4f}")
print(f"Robustness Gap: {robustness_gap:.4f}")


# In[73]:


print ("MIM attempt")
acc_clean = evaluate(params, x_test, y_test)
print(f"Clean accuracy: {acc_clean:.4f}")

x_adv = MIM(model_fn, params, x_test, y_test, eps=0.2)
acc_adv = evaluate(params, x_adv, y_test)

asr = 1.0 - acc_adv
robustness_gap = acc_clean - acc_adv

print(f"Adversarial accuracy: {acc_adv:.4f}")
print(f"Attack Success Rate: {asr:.4f}")
print(f"Robustness Gap: {robustness_gap:.4f}")


# In[74]:


print ("PGD attempt")
acc_clean = evaluate(params, x_test, y_test)
print(f"Clean accuracy: {acc_clean:.4f}")

x_adv = PGD(model_fn, params, x_test, y_test, eps=0.2)
acc_adv = evaluate(params, x_adv, y_test)

asr = 1.0 - acc_adv
robustness_gap = acc_clean - acc_adv

print(f"Adversarial accuracy: {acc_adv:.4f}")
print(f"Attack Success Rate: {asr:.4f}")
print(f"Robustness Gap: {robustness_gap:.4f}")


# In[75]:


print ("SA attempt")
acc_clean = evaluate(params, x_test, y_test)
print(f"Clean accuracy: {acc_clean:.4f}")
x_adv = SA(model_fn, x_test, y_test, eps=0.2)
acc_adv = evaluate(params, x_adv, y_test)

asr = 1.0 - acc_adv
robustness_gap = acc_clean - acc_adv

print(f"Adversarial accuracy: {acc_adv:.4f}")
print(f"Attack Success Rate: {asr:.4f}")
print(f"Robustness Gap: {robustness_gap:.4f}")


# In[76]:


print ("APGD attempt")
acc_clean = evaluate(params, x_test, y_test)
print(f"Clean accuracy: {acc_clean:.4f}")
#def     APGD(model_fn, params, x, y, eps=0.3, alpha=0.01, num_steps=10, seed=0)
x_adv = APGD(model_fn, params, x_test, y_test, steps=3)
acc_adv = evaluate(params, x_adv, y_test)

asr = 1.0 - acc_adv
robustness_gap = acc_clean - acc_adv

print(f"Adversarial accuracy: {acc_adv:.4f}")
print(f"Attack Success Rate: {asr:.4f}")
print(f"Robustness Gap: {robustness_gap:.4f}")

