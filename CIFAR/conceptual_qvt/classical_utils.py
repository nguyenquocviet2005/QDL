import torch
import torch.nn as nn

def split_into_patches(image_tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Splits a batch of images into patches.
    Args:
        image_tensor: Tensor of shape (batch_size, channels, height, width)
        patch_size: The height and width of each square patch.
    Returns:
        Tensor of patches: (batch_size, num_patches, patch_size*patch_size*channels)
                         or (batch_size, num_patches, patch_dim)
    """
    batch_size, channels, height, width = image_tensor.shape
    assert height % patch_size == 0 and width % patch_size == 0, \
        f"Image dimensions ({height}x{width}) must be divisible by patch size ({patch_size})."

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w

    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches shape: (batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)
    
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # patches shape: (batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
    
    patches = patches.view(batch_size, num_patches, -1)
    # patches shape: (batch_size, num_patches, channels*patch_size*patch_size)
    return patches

class PatchEmbed(nn.Module):
    """
    Linearly embeds image patches and prepends a CLS token.
    """
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection for patches
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)
        
        # Learnable CLS token
        # Initialize with zeros or small random values, consistent with ViT practice
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
        Returns:
            Tensor of shape (batch_size, num_patches + 1, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match model's expected size ({self.img_size}x{self.img_size})."

        patches = split_into_patches(x, self.patch_size)  # (B, num_patches, patch_dim)
        projected_patches = self.proj(patches)            # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        embeddings = torch.cat((cls_tokens, projected_patches), dim=1) # (B, num_patches + 1, embed_dim)
        return embeddings 