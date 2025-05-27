import torch
import torch.nn as nn
import math
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np # For random subset selection

# Assuming quantum_operations.py and classical_utils.py are in the same directory or accessible
from quantum_operations import (
    concept_d_encode, concept_a_encode, build_d_encoding,
    qram_store, perform_tomography,
    quantum_add, quantum_normalize, quantum_linear,
    quantum_matmul, quantum_softmax, quantum_relu, qdac_d_to_a
)
from classical_utils import PatchEmbed

class QPos(nn.Module):
    """Quantum Positional Encoding"""
    def __init__(self, embed_dim: int, max_seq_len: int = 257): # e.g., 16x16 patches for 256x256 img -> 256 patches + 1 CLS
        super().__init__()
        self.embed_dim = embed_dim
        # P ∈ R^(d x n) in paper. Here (max_seq_len, embed_dim) for easier indexing.
        # Stored as classical nn.Parameter, conceptually D-Encoded when used.
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02) # Learnable

    def __call__(self, embeddings_d_encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_d_encoded: D-Encoded input embeddings (batch_size, seq_len, embed_dim)
        Returns:
            D-Encoded embeddings with positional information added.
        """
        # D-Encode positional embeddings (conceptually, as they are classical params)
        # The input embeddings_d_encoded are already D-Encoded.
        # The stored self.pos_embedding is classical, needs conceptual D-Encoding.
        # The operation is X_out = X_in + P.
        # P needs to match batch size and current seq_len of x.
        batch_size, seq_len, _ = embeddings_d_encoded.shape
        # Use part of pos_embedding corresponding to current sequence length
        pos_embed_to_add = self.pos_embedding[:, :seq_len, :]
        
        # Conceptually, pos_embed_to_add is D-Encoded before quantum_add
        conceptual_pos_embed_d_encoded = concept_d_encode(pos_embed_to_add.expand(batch_size, -1, -1))
        
        # Perform quantum addition (element-wise for simulation)
        output_d_encoded = quantum_add(embeddings_d_encoded, conceptual_pos_embed_d_encoded)
        return output_d_encoded

class QNorm(nn.Module):
    """Quantum Layer Normalization"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # QNorm doesn't have learnable gamma/beta like classical LayerNorm in this paper's description

    def __call__(self, state_d_encoded: torch.Tensor) -> torch.Tensor:
        return quantum_normalize(state_d_encoded, self.embed_dim)

class QAdd(nn.Module):
    """Quantum Add for Residual Connections"""
    def __init__(self):
        super().__init__()

    def __call__(self, state1_d_encoded: torch.Tensor, state2_d_encoded: torch.Tensor) -> torch.Tensor:
        # Ensures both inputs are D-Encoded as per paper (tomography before QAdd for one branch)
        return quantum_add(state1_d_encoded, state2_d_encoded)

class QAttn(nn.Module):
    """Quantum Multi-Head Self-Attention"""
    def __init__(self, embed_dim: int, num_heads: int, tomography_error_delta: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tomography_error_delta = tomography_error_delta

        # Classical learnable parameters for Q, K, V, and output projection
        # W_qm, W_km, W_vm ∈ R^(d x d) in paper. Here, we combine them for all heads.
        # So, W_q, W_k, W_v are (embed_dim, embed_dim)
        self.W_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(embed_dim, embed_dim))
        # W_o in pseudocode: W ∈ R^(d x hd). Here, hd is embed_dim.
        self.W_o = nn.Parameter(torch.randn(embed_dim, embed_dim)) 

        # Initialize weights (e.g., Xavier uniform)
        for param in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(param)

    def __call__(self, x_d_encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_d_encoded: Input D-Encoded state (batch_size, seq_len, embed_dim)
        Returns:
            Output D-Encoded state after attention and tomography.
        """
        B, N, C = x_d_encoded.shape # Batch size, Sequence length, Embedding dimension
        
        # 1. Compute Q, K, V using quantum_linear (conceptually on D-Encoded input)
        # Q_m = W_qm X, K_m = W_km X, V_m = W_vm X
        # These W are (embed_dim, embed_dim). Output is (B, N, C)
        q_d_encoded = quantum_linear(x_d_encoded, self.W_q) # (B, N, C)
        k_d_encoded = quantum_linear(x_d_encoded, self.W_k) # (B, N, C)
        v_d_encoded = quantum_linear(x_d_encoded, self.W_v) # (B, N, C)

        # Reshape for multi-head: (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q_multi_head = q_d_encoded.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_multi_head = k_d_encoded.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_multi_head = v_d_encoded.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. Compute Attention Scores: A_m = K_m^T Q_m
        # (B, num_heads, head_dim, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, N)
        # Note: The paper shows A_m = K_m^T Q_m. If Q,K are (d,n), then K^T Q is (n,d)x(d,n) = (n,n)
        # For our (B, num_heads, N, head_dim), K.transpose is (B, num_heads, head_dim, N)
        # So, attn_scores = q_multi_head @ k_multi_head.transpose(-2, -1)
        # Or, if we follow K^T Q literally with N=seq_len, C_head=head_dim:
        # Q: (B, H, N, C_h), K: (B, H, N, C_h) -> K.T: (B, H, C_h, N)
        # K.T @ Q -> (B, H, C_h, C_h). This doesn't seem right for typical attention scores (N,N).
        # Standard attention: (Q @ K.T) / sqrt(d_k)
        # Let's use the standard (Q K^T)
        attn_scores_encoded = quantum_matmul(q_multi_head, k_multi_head.transpose(-2, -1)) # (B, num_heads, N, N)
        
        # 3. Scale: A_m / sqrt(d_k) (d_k is head_dim)
        attn_scores_scaled_encoded = attn_scores_encoded / math.sqrt(self.head_dim)
        
        # 4. Apply Softmax: A_m' = softmax(A_m / sqrt(d_k)), column-wise (last dim)
        # Pseudocode: quantum_softmax(attn_scores) -> A-Encoding
        attn_weights_a_encoded = quantum_softmax(attn_scores_scaled_encoded, dim=-1) # (B, num_heads, N, N)
        # This is conceptually A-Encoded according to pseudocode comment

        # 5. Compute Head Output: H_m = V_m A_m'
        # (B, num_heads, N, head_dim) @ (B, num_heads, N, N) -- this is not V A, but A V
        # Standard: attn_weights @ V
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        head_outputs_a_encoded = quantum_matmul(attn_weights_a_encoded, v_multi_head) # (B, num_heads, N, head_dim)
        # This is also conceptually A-Encoded

        # 6. Concatenate Heads: H = Concat(H_0, ..., H_{h-1})
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        concatenated_heads_a_encoded = head_outputs_a_encoded.permute(0, 2, 1, 3).contiguous().view(B, N, C)

        # 7. Project: X^out = W_o H
        # W_o is (embed_dim, embed_dim). Input H is (B, N, C=embed_dim)
        # Output is (B, N, C). Pseudocode: quantum_linear(head_outputs, self.W_o) -> A-Encoding
        # Input to quantum_linear here is concatenated_heads_a_encoded
        output_a_encoded = quantum_linear(concatenated_heads_a_encoded, self.W_o)
        # This is A-Encoded.

        # 8. Perform Quantum Tomography (QTomo) with l_∞ tomography
        classical_output = perform_tomography(output_a_encoded, self.tomography_error_delta)

        # 9. Store results in qRAM ("Store & Reuse") - conceptual
        qram_store(classical_output, name="QAttn_output_after_tomo")

        # 10. Reconstruct D-Encoding from tomography results.
        output_d_encoded = build_d_encoding(classical_output)
        
        return output_d_encoded

class QFFN(nn.Module):
    """Quantum Feed-Forward Network"""
    def __init__(self, embed_dim: int, ffn_hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        # Classical learnable parameters
        self.W_1 = nn.Parameter(torch.randn(ffn_hidden_dim, embed_dim))
        self.b_1 = nn.Parameter(torch.randn(ffn_hidden_dim))
        self.W_2 = nn.Parameter(torch.randn(embed_dim, ffn_hidden_dim))
        self.b_2 = nn.Parameter(torch.randn(embed_dim))

        # Initialize
        nn.init.xavier_uniform_(self.W_1)
        nn.init.zeros_(self.b_1)
        nn.init.xavier_uniform_(self.W_2)
        nn.init.zeros_(self.b_2)

    def __call__(self, x_d_encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_d_encoded: Input D-Encoded state (batch_size, seq_len, embed_dim)
        Returns:
            Output D-Encoded state.
        """
        # X^mid = W_1 X^in + b_1 (D-Encoding)
        mid_d_encoded = quantum_linear(x_d_encoded, self.W_1, self.b_1)
        
        # Apply ReLU: f(X^mid) (D-Encoding)
        mid_relu_d_encoded = quantum_relu(mid_d_encoded)
        
        # X^out = W_2 f(X^mid) + b_2 (D-Encoding)
        output_d_encoded = quantum_linear(mid_relu_d_encoded, self.W_2, self.b_2)
        
        return output_d_encoded

class QEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int, tomography_error_delta: float = 0.0):
        super().__init__()
        self.qnorm1 = QNorm(embed_dim)
        self.qattn = QAttn(embed_dim, num_heads, tomography_error_delta=tomography_error_delta)
        self.qadd1 = QAdd()
        
        self.qnorm2 = QNorm(embed_dim)
        self.qffn = QFFN(embed_dim, ffn_hidden_dim)
        self.qadd2 = QAdd()

    def __call__(self, quantum_state_d_encoded: torch.Tensor) -> torch.Tensor:
        # Input is D-Encoded
        # First block: QNorm -> QAttn -> QAdd (with residual)
        # quantum_state_d_encoded is X^in for the first residual connection
        
        norm_state_d_encoded = self.qnorm1(quantum_state_d_encoded) # D-Encoding
        
        # QAttn output is D-Encoding after tomography and reconstruction
        attn_state_d_encoded = self.qattn(norm_state_d_encoded) # D-Encoding
        
        # QAdd: X^out = X^QAttn + X^in (Quantum Residual Connection)
        # Paper: "QTomo before QAdd ensures residual input preservation"
        # attn_state_d_encoded is the result after tomography.
        # quantum_state_d_encoded is the original input to QNorm (X^in to the layer).
        added_state_d_encoded = self.qadd1(attn_state_d_encoded, quantum_state_d_encoded) # D-Encoding
        
        # Second block: QNorm -> QFFN -> QAdd (with residual)
        # added_state_d_encoded is X^in for the second residual connection
        
        norm_state2_d_encoded = self.qnorm2(added_state_d_encoded) # D-Encoding
        ffn_state_d_encoded = self.qffn(norm_state2_d_encoded)     # D-Encoding
        
        # QAdd: X^out = X^QFFN + X^in (input to second QNorm)
        final_state_d_encoded = self.qadd2(ffn_state_d_encoded, added_state_d_encoded) # D-Encoding
        return final_state_d_encoded

class QHead(nn.Module):
    """Quantum Classification Head"""
    def __init__(self, embed_dim: int, num_classes: int, tomography_error_delta: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tomography_error_delta = tomography_error_delta

        # Classical learnable parameters: W ∈ R^(Kxd), b ∈ R^K
        self.W = nn.Parameter(torch.randn(num_classes, embed_dim))
        self.b = nn.Parameter(torch.randn(num_classes))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def __call__(self, cls_token_d_encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token_d_encoded: D-Encoding of the CLS token (batch_size, 1, embed_dim)
                                  or (batch_size, embed_dim) if squeezed.
        Returns:
            Classical classification scores (batch_size, num_classes).
        """
        if cls_token_d_encoded.ndim == 3 and cls_token_d_encoded.shape[1] == 1:
            cls_token_d_encoded = cls_token_d_encoded.squeeze(1) # (batch_size, embed_dim)
        
        # X^out = W x_0^in + b
        # quantum_linear input is D-Encoded, output is D-Encoded conceptually
        output_d_encoded = quantum_linear(cls_token_d_encoded, self.W, self.b) # (batch_size, num_classes)
        
        # Output as A-Encoding: |X^out⟩ (via QDAC)
        output_a_encoded = qdac_d_to_a(output_d_encoded)
        
        # Perform QTomo to sample classical X^out
        classical_scores = perform_tomography(output_a_encoded, self.tomography_error_delta)
        
        return classical_scores

class QViT(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, num_classes: int,
                 embed_dim: int = 768, num_layers: int = 12, num_heads: int = 12, 
                 ffn_hidden_dim: int = 3072, tomography_error_delta: float = 0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.tomography_error_delta = tomography_error_delta

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        max_seq_len = num_patches + 1 # CLS token
        
        self.qpos = QPos(embed_dim, max_seq_len=max_seq_len)
        
        self.encoders = nn.ModuleList([
            QEncoderLayer(embed_dim, num_heads, ffn_hidden_dim, tomography_error_delta)
            for _ in range(num_layers)
        ])
        
        self.qhead = QHead(embed_dim, num_classes, tomography_error_delta)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor: Input image (batch_size, channels, height, width)
        Returns:
            Classical classification scores (batch_size, num_classes)
        """
        # 1. Classical Preprocessing & Embedding
        # Output: (batch_size, num_patches + 1, embed_dim)
        classical_embeddings = self.patch_embed(image_tensor)

        # 2. Quantum Encoding (D-Encoding)
        # The resulting vectors (X ∈ R^(d x n)) are encoded into quantum states using D-Encoding
        # X^in = classical_embeddings
        current_state_d_encoded = concept_d_encode(classical_embeddings)
        
        # 3. Add Positional Embeddings (Quantum)
        # X^out = X^in + P (D-Encoding output)
        current_state_d_encoded = self.qpos(current_state_d_encoded)
        
        # 4. Quantum Transformer Encoder Layers
        for encoder_layer in self.encoders:
            current_state_d_encoded = encoder_layer(current_state_d_encoded) # D-Encoding
        
        # 5. Quantum Classification Head
        # Input: D-Encoding of the CLS token from the last encoder layer (x_0^in ∈ R^d)
        cls_token_d_encoded = current_state_d_encoded[:, 0, :] # (batch_size, embed_dim)
        # Add a dim for seq_len=1 if QHead expects (B,1,C)
        # cls_token_d_encoded = current_state_d_encoded[:, 0, :].unsqueeze(1) 
        
        # QHead takes D-Encoded CLS token, performs linear, QDAC to A-Encoding, then Tomography
        classical_output_scores = self.qhead(cls_token_d_encoded)
        
        # label = torch.argmax(classical_output_scores, dim=-1) # If returning labels
        return classical_output_scores


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CIFAR-10 Data Loading and Preprocessing ---
    IMG_SIZE_CIFAR10 = 32
    TRAIN_SIZE = 1000
    VAL_SIZE = 1000
    BATCH_SIZE = 32 # Smaller batch size for potentially large models
    NUM_EPOCHS = 100 # For demonstration
    LR = 1e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 Normalize
    ])

    # Load full CIFAR-10 training and test sets
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create random indices for subsets
    total_train_size = len(full_train_dataset)
    indices = list(range(total_train_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:TRAIN_SIZE]
    # Use a different part of the full training set for validation to avoid overlap with potential test set usage
    # Or split the test_dataset if preferred. Here, using a portion of the original training data for validation.
    if TRAIN_SIZE + VAL_SIZE <= total_train_size:
        val_indices = indices[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    else: # Fallback if requested sizes are too large, use part of test set for val or reduce val_size
        print(f"Warning: TRAIN_SIZE + VAL_SIZE > total_train_data. Adjusting VAL_SIZE or using test set for validation.")
        # Simplified: just take from test_set for this example if overflow
        val_indices = list(range(len(test_dataset)))
        np.random.shuffle(val_indices)
        val_indices = val_indices[:VAL_SIZE]
        val_dataset = Subset(test_dataset, val_indices)


    train_subset = Subset(full_train_dataset, train_indices)
    if 'val_dataset' not in locals(): # If val_indices came from full_train_dataset
        val_subset = Subset(full_train_dataset, val_indices)
    else: # If val_dataset was created from test_dataset
        val_subset = val_dataset


    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) # Full test set

    print(f"Training with {len(train_subset)} images, validating with {len(val_subset)} images.")

    # --- QViT Model Instantiation for CIFAR-10 ---
    qvit_model_cifar10 = QViT(
        img_size=IMG_SIZE_CIFAR10,      # 32 for CIFAR-10
        patch_size=4,                   # 4x4 patches -> 8x8 grid = 64 patches
        in_chans=3,
        num_classes=10,
        embed_dim=192,                  # Smaller embedding dimension
        num_layers=6,                   # Fewer layers
        num_heads=3,                    # Fewer heads (192 % 3 == 0)
        ffn_hidden_dim=192 * 4,         # Standard FFN scaling
        tomography_error_delta=0.001    # Example error
    ).to(device)

    print(f"QViT Model for CIFAR-10 Instantiated.")
    total_params = sum(p.numel() for p in qvit_model_cifar10.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(qvit_model_cifar10.parameters(), lr=LR, weight_decay=0.01)
    
    # --- Training and Evaluation Loop ---
    for epoch in range(NUM_EPOCHS):
        # Training phase
        qvit_model_cifar10.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = qvit_model_cifar10(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train

        # Validation phase
        qvit_model_cifar10.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = qvit_model_cifar10(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    print("Training finished.")

    # Optional: Final evaluation on the full test set
    # qvit_model_cifar10.eval()
    # correct_test = 0
    # total_test = 0
    # test_loss = 0
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = qvit_model_cifar10(images)
    #         loss = criterion(outputs, labels)
    #         test_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total_test += labels.size(0)
    #         correct_test += (predicted == labels).sum().item()
    # final_test_loss = test_loss / len(test_loader)
    # final_test_acc = 100 * correct_test / total_test
    # print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Acc: {final_test_acc:.2f}%")

# Original example usage (commented out or removed for CIFAR-10 focus)
# if __name__ == '__main__':
#     # Example Usage (requires PyTorch)
#     bs = 2
#     img_s = 256
#     patch_s = 16 # ViT-Base uses 16x16 patches
#     in_c = 3
#     num_cls = 10 # Example number of classes
    
#     emb_d = 768       # ViT-Base: 768
#     n_layers = 12     # ViT-Base: 12
#     n_heads = 12      # ViT-Base: 12
#     ffn_hid_d = 3072  # ViT-Base: 4 * embed_dim = 3072
#     tomo_err = 0.001  # Example tomography error

#     # Create a dummy image tensor
#     dummy_image = torch.randn(bs, in_c, img_s, img_s)

#     # Instantiate QViT
#     qvit_model = QViT(
#         img_size=img_s, 
#         patch_size=patch_s, 
#         in_chans=in_c, 
#         num_classes=num_cls,
#         embed_dim=emb_d,
#         num_layers=n_layers,
#         num_heads=n_heads,
#         ffn_hidden_dim=ffn_hid_d,
#         tomography_error_delta=tomo_err
#     )

#     print(f"QViT Model Instantiated: {qvit_model}\\n")

#     # Forward pass
#     print("Performing forward pass...")
#     try:
#         output_scores = qvit_model(dummy_image)
#         print(f"\\nOutput scores shape: {output_scores.shape}") # Expected: (batch_size, num_classes)
#         print(f"Output scores:\\n{output_scores}")
#     except Exception as e:
#         print(f"Error during forward pass: {e}")
#         import traceback
#         traceback.print_exc()

#     print("\\nChecking parameter counts...")
#     total_params = sum(p.numel() for p in qvit_model.parameters() if p.requires_grad)
#     print(f"Total trainable parameters in QViT (classical part): {total_params:,}")

#     # For comparison, a classical ViT-Base has ~86M parameters.
#     # This count is only for the PyTorch nn.Parameters. The "quantumness" is in the ops.

#     # Example: Accessing CLS token from patch_embed
#     # print("\\nCLS token from patch_embed:", qvit_model.patch_embed.cls_token)
#     # Example: Accessing positional embeddings from qpos
#     # print("Positional embeddings from qpos:", qvit_model.qpos.pos_embedding.shape) 