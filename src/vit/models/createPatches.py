import torch.nn as nn
import torch


class CreatePatches(nn.Module):
    def __init__(self, channels: int = 3, embed_dim: int = 768, patch_size: int = 16):
        super().__init__()
        # kernel & stride both have size `patch_size`
        self.patch = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x, verbose=False):
        patches = self.patch(x)  # Shape: B x out_channels x ( H / patch_size)
        if verbose:
            print(f'patches shape: {patches.shape}')

        # Flatten along dim = 2 to maintain channel dimension.
        patches = patches.flatten(2).transpose(1, 2)  # Shape: B x patch_size x out_channels
        if verbose:
            print(f'patches shape after flatten & transpose: {patches.shape}')
        return patches


if __name__ == '__main__':

    channels = 3
    embed_dim = 768
    patch_size = 16
    batch_size = 1
    height, width = 64, 64

    assert height % patch_size == 0 and width % patch_size == 0

    model = CreatePatches(channels=channels, embed_dim=embed_dim, patch_size=patch_size)

    # Create random Tensor with shape B x C x H x W
    x = torch.randn(batch_size, channels, height, width)
    print(f'Shape of x: {x.shape}')
    patches = model(x)
    print(f'Shape of patches: {patches.shape}')

    expected_shape = (batch_size, (height // patch_size) * (width // patch_size), embed_dim)
    print(f'Expected shape: {expected_shape}')
    assert patches.shape == expected_shape, f"Expected shape: {expected_shape}, Actual shape: {patches.shape}"

    print('Test Passed')
