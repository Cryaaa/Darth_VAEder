# %%
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from darth_vaeder.datamodules.dataset_EG import percentile_norm
from darth_vaeder.models.lit_vae import LitVAE

# %%
CHECKPOINT = "/mnt/efs/dl_jrc/student_data/S-EG/project/VAE1/logs/vae/version_38/checkpoints/last.ckpt"
FACES_DIR = pathlib.Path("/mnt/efs/dl_jrc/data/representation_faces/crop")
OUTPUT_DIR = pathlib.Path("/mnt/efs/dl_jrc/student_data/S-EG/project/Darth_VAEder/notebooks/face_reconstructions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LitVAE.load_from_checkpoint(CHECKPOINT, map_location=device, weights_only=False).eval().to(device)


# %%
def load_face(path: pathlib.Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0  # (H, W, 3)
    img = np.transpose(img, (2, 0, 1))  # (3, H, W)
    r = percentile_norm(img[0])
    g = percentile_norm(img[1])
    b = percentile_norm(img[2])
    return np.stack([r, g, b], axis=0).astype(np.float32)  # (3, H, W)


# %%
face_paths = sorted(FACES_DIR.glob("*.png"))
face_paths
# %%
from skimage import io

io.imshow(load_face(face_paths[0])[2])
# %%
results: dict[str, dict] = {}

with torch.no_grad():
    for path in face_paths:
        inp = load_face(path)
        x = torch.tensor(inp).unsqueeze(0).to(device)
        recon, z, mu, logvar = model.vae(x)
        results[path.stem] = {
            "inp": inp,
            "recon": recon.squeeze(0).cpu().numpy(),
            "mu": mu.squeeze(0).cpu().numpy(),
        }

# %%
n = len(results)
fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
if n == 1:
    axes = axes[np.newaxis]

for row, (name, data) in zip(axes, results.items()):
    inp = data["inp"]
    recon = data["recon"]

    row[0].imshow(inp[0], cmap="gray", vmin=0, vmax=1)
    row[0].set_title(f"{name}\ninput ch0 (R)")
    row[0].axis("off")

    row[1].imshow(inp[1], cmap="gray", vmin=0, vmax=1)
    row[1].set_title(f"{name}\ninput ch1 (G)")
    row[1].axis("off")

    row[2].imshow(np.clip(recon[0], 0, 1), cmap="gray")
    row[2].set_title(f"{name}\nrecon ch0")
    row[2].axis("off")

    row[3].imshow(np.clip(recon[1], 0, 1), cmap="gray")
    row[3].set_title(f"{name}\nrecon ch1")
    row[3].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "all_faces.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
for name, data in results.items():
    inp = data["inp"]
    recon = data["recon"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    axes[0, 0].imshow(inp[0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("input ch0 (R norm)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(inp[1], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("input ch1 (G norm)")
    axes[0, 1].axis("off")

    inp_rgb = np.stack([inp[0], inp[1], np.zeros_like(inp[0])], axis=-1)
    axes[0, 2].imshow(np.clip(inp_rgb, 0, 1))
    axes[0, 2].set_title("input RG composite")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(np.clip(recon[0], 0, 1), cmap="gray")
    axes[1, 0].set_title("recon ch0")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.clip(recon[1], 0, 1), cmap="gray")
    axes[1, 1].set_title("recon ch1")
    axes[1, 1].axis("off")

    recon_rgb = np.stack([np.clip(recon[0], 0, 1), np.clip(recon[1], 0, 1), np.zeros_like(recon[0])], axis=-1)
    axes[1, 2].imshow(np.clip(recon_rgb, 0, 1))
    axes[1, 2].set_title("recon RG composite")
    axes[1, 2].axis("off")

    fig.suptitle(name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
