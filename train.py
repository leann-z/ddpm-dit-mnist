import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import math
import numpy as np
import einops
from pathlib import Path
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from DIT import DiT

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 256
DEPTH = 4
HEADS = 4
MLP_RATIO = 4.0
CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 40
LR = 3e-4
T = 500
device = "mps"


# ============================================================================
# DDPM UTILS
# ============================================================================


def get_ddpm_schedule(T):
    betas = torch.linspace(1e-4, 0.03, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def forward_diffusion(x0, t, alphas_cumprod):
    ### YOUR CODE STARTS HERE ###
    # Your code should compute xt at timestep t, and also return the noise
    noise = torch.randn_like(x0)

    abar_t = alphas_cumprod[t].view(-1, 1, 1, 1)

    xt = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * noise
    
    ### YOUR CODE ENDS HERE ###
    return xt, noise


@torch.no_grad()
def sample_ddpm(net, T, bsz, betas, alphas, alphas_cumprod, num_snapshots=10, y=None, cfg_scale=None):
    net.eval()

    # sample the initial noise
    x = torch.randn(bsz, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # Identify which timesteps to save for the visualization grid
    snapshot_indices = torch.linspace(T - 1, 0, num_snapshots).long()
    snapshots = []

    for t in reversed(range(T)):

        ### YOUR CODE STARTS HERE ###
        t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)

        if cfg_scale is not None and y is not None:
            # use CFG
            eps_pred = net.forward_with_cfg(x, t_batch, y, cfg_scale)
        elif y is not None:
            y = y.to(device)
            eps_pred = net(x, t_batch, y=y)
        else:
            eps_pred = net(x, t_batch)
        

        beta_t = betas[t]
        alpha_t = alphas[t]
        abar_t = alphas_cumprod[t]

        # denoising
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - abar_t)
        mean = coef1 * (x - coef2 * eps_pred)

        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z
        else:
            x = mean
        # Your code should compute xt at timestep t
        ### YOUR CODE ENDS HERE ###

        if t in snapshot_indices:
            snapshots.append(x.cpu())

    # Return shape: (num_snapshots, bsz, C, H, W)
    return torch.stack(snapshots)


def visualize_forward_diffusion(dataloader, alphas_cumprod, n_steps=10):
    # Get a batch of real images
    images, _ = next(iter(dataloader))
    images = images[:8]

    # Select timesteps to show (0 to T-1)
    indices = torch.linspace(0, T - 1, n_steps).long()

    cols = []
    for t in indices:
        t_batch = torch.full((images.shape[0],), t, dtype=torch.long)
        # Apply the forward diffusion
        xt, _ = forward_diffusion(images.to(device), t_batch.to(device), alphas_cumprod)
        cols.append(xt.cpu())

    # Stack and rearrange: (Steps, Batch, C, H, W) -> (Batch * Steps, C, H, W)
    result = torch.stack(cols, dim=1)
    result = einops.rearrange(result, "b t c h w -> (b t) c h w")

    grid = vutils.make_grid(result, nrow=n_steps, normalize=True, value_range=(-1, 1))

    plt.figure(figsize=(15, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.savefig("images/forward_diffusion_process.png")
    plt.show()

def visualize_attention_maps(net, dataloader, betas, alphas, alphas_cumprod, T, device="mps"):
    net.eval()
    
    attention_maps = {}
    denoised_images = {}
    
    def make_hook(timestep_label):
        def hook_fn(module, input, output):
            x = input[0]
            B, N, D = x.shape
            num_heads = module.num_heads
            head_dim = D // num_heads
            
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attention_maps[timestep_label] = attn[0].mean(0).detach().cpu()
        return hook_fn
    
    probe_timesteps = [10, 100, 250, 400, 490]
    digit = 7
    y = torch.full((1,), digit, device=device, dtype=torch.long)
    x = torch.randn(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    
    with torch.no_grad():
        for t_val in reversed(range(T)):
            t_batch = torch.full((1,), t_val, device=device, dtype=torch.long)
            
            if t_val in probe_timesteps:
                hook = net.blocks[0].attn.register_forward_hook(make_hook(t_val))
            
            eps_pred = net(x, t_batch, y=y)
            
            if t_val in probe_timesteps:
                hook.remove()
            
            beta_t = betas[t_val]
            alpha_t = alphas[t_val]
            abar_t = alphas_cumprod[t_val]
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - abar_t)) * eps_pred
            )
            if t_val > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
            
            if t_val in probe_timesteps:
                denoised_images[t_val] = x[0, 0].detach().cpu()

    n_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
    grid_size = IMAGE_SIZE // PATCH_SIZE
    centre_patch = (n_patches // 2) + (grid_size // 2)

    fig, axes = plt.subplots(2, len(probe_timesteps), figsize=(15, 6))
    
    for idx, t_val in enumerate(probe_timesteps):
        # curr image at timestep
        img = denoised_images[t_val].numpy()
        axes[0, idx].imshow(img, cmap='gray', aspect='auto')
        axes[0, idx].set_title(f't = {t_val}', fontsize=11)
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_ylabel('Denoised Image', fontsize=9)

        # spatial attention from center patch
        attn = attention_maps[t_val]
        spatial_attn = attn[centre_patch].reshape(grid_size, grid_size).numpy()
        axes[1, idx].imshow(spatial_attn, cmap='viridis', aspect='auto')
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_ylabel('Spatial Attention\n(from centre patch)', fontsize=9)
    
    plt.suptitle(f'DiT Attention Maps During Denoising (digit={digit})', fontsize=12)
    plt.tight_layout()
    plt.savefig('images/attention_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved images/attention_maps.png")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    Path("images").mkdir(exist_ok=True)

    # 1. Data Setup
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # 2. Model Setup
    net = DiT(
        input_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        hidden_size=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        mlp_ratio=MLP_RATIO,
        num_classes=10,
        learn_sigma=False,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

    optimizer = optim.AdamW(net.parameters(), lr=LR)
    # get noise schedule parameters
    betas, alphas, alphas_cumprod = get_ddpm_schedule(T)

    # Visualize the forward process before training
    visualize_forward_diffusion(dataloader, alphas_cumprod)
    
    # 3. Training Loop
    print("Starting Training...")
    loss_history = []

    for epoch in range(EPOCHS):
        net.train()
        epoch_loss = 0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, T, (bsz,), device=device).long()

            # Add noise
            xt, noise = forward_diffusion(images, t, alphas_cumprod)

            # Predict noise (part b rn)
            #remove y = labels if part a
            pred_noise = net(xt, t, y=labels)

            # Use MSE loss
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

        # Inside the training loop:
        if (epoch + 5) % 5 == 0:
            num_samples = 8
            num_steps = 10  # How many steps of the process to show
            y_sample = torch.arange(8, device=device)

            # Generate the trajectory
            # Shape: (num_steps, num_samples, C, H, W)
            traj = sample_ddpm(
                net, T, num_samples, betas, alphas, alphas_cumprod, num_steps, y=y_sample, cfg_scale=5.0
            )

            grid_ready = einops.rearrange(traj, "s b c h w -> b s c h w")
            grid_ready = einops.rearrange(grid_ready, "b s c h w -> (b s) c h w")

            # Create the grid
            grid = vutils.make_grid(
                grid_ready, nrow=num_steps, normalize=True, value_range=(-1, 1)
            )

            # Save the result
            vutils.save_image(grid, f"images/evolution_epoch_{epoch+1}.png")
            print(
                f"Epoch {epoch+1}: Saved generation grid to images/evolution_epoch_{epoch+1}.png"
            )


    visualize_attention_maps(net, dataloader, betas, alphas, alphas_cumprod, T, device=device)

    # part c sampling
    cfg_scales = [1, 5, 10, 15, 20]

    for cfg in cfg_scales:
        all_imgs = []
        for digit in range(10):
            y = torch.full((5,), digit, device=device, dtype=torch.long)
            traj = sample_ddpm(net, T, 5, betas, alphas, alphas_cumprod, num_snapshots=10, y=y, cfg_scale=cfg)
            all_imgs.append(traj[-1].cpu())
        
        all_imgs = torch.cat(all_imgs, dim=0)
        grid = vutils.make_grid(all_imgs, nrow=5, normalize=True, value_range=(-1, 1))
        vutils.save_image(grid, f"images/cfg_scale_{cfg}.png")
        print(f"Saved cfg_scale_{cfg}.png")
    

    # Save Loss Plot
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.savefig("images/loss_curve.png")

    print("Done! Check the 'images' folder.")
