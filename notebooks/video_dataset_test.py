# %%
import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import zarr
from matplotlib.collections import LineCollection
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from umap import UMAP

from darth_vaeder.datamodules.dataset_EG import BCDataModule, BorderCellDataset, no_transform, percentile_norm
from darth_vaeder.models.lit_vae import LitVAE

# %%
CHECKPOINT = "/mnt/efs/dl_jrc/student_data/S-EG/project/VAE1/logs/vae/version_38/checkpoints/last.ckpt"
METADATA_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x_with_morphology.csv"
OUTPUT_CSV_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x_with_LatentsAndMorphology.csv"
METADATA_PATH_VIDEOS = "/mnt/efs/dl_jrc/student_data/S-EG/cropping_emmily/data_information_videos.csv"
ZARR_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/cropping_emmily/border_cell_videos.zarr"
OUTPUT_ROOT = pathlib.Path("/mnt/efs/dl_jrc/student_data/S-EG/project/Darth_VAEder/notebooks/trajectory_frames")
CMAP = plt.cm.viridis


# %%
class VideoFrameDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        zarr_path,
        input_array_name,
        input_mask_name,
        channels=[0, 2],
        spatial_transforms=no_transform,
        intensity_transforms=no_transform,
        normalization_function=percentile_norm,
    ):
        self.metadata = pd.read_csv(annotations_file).reset_index(drop=True)
        zarr_group = zarr.open(zarr_path)
        self.spatial_transforms = spatial_transforms
        self.intensity_transforms = intensity_transforms

        self.inputs = []
        self.masks = []
        for _, row in self.metadata.iterrows():
            sample_name = str(row["sample_name"])
            frame_index = int(row["frame_index"])

            video_group = zarr_group[sample_name]
            frame = np.array(video_group[input_array_name][frame_index])

            if frame.ndim == 2:
                frame = frame[np.newaxis]

            frame = frame[channels]
            normalized_channels = [normalization_function(ch) for ch in frame]
            frame = np.stack(normalized_channels, axis=0).astype(np.float32)

            mask = np.array(video_group[input_mask_name][frame_index])
            mask = np.max(mask * 1.0, axis=(0, 1), keepdims=True).astype(np.float32)[0]
            self.inputs.append(frame)
            self.masks.append(mask)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputsT = torch.tensor(self.inputs[idx])
        masksT = torch.tensor(self.masks[idx])
        targetT = torch.tensor(self.inputs[idx])

        inputsT = self.spatial_transforms(inputsT)
        masksT = self.spatial_transforms(masksT)
        targetT = self.spatial_transforms(targetT)

        inputsT = self.intensity_transforms(inputsT)

        return {
            "source": inputsT,
            "target": targetT,
            "masks": masksT,
            "metadata_index": self.metadata.iloc[idx]["image_id"],
        }


def get_latent_features(model, loader):
    model.eval()
    latents = []
    logvars = []
    recons = []
    metadata_list = []

    with torch.no_grad():
        for batch in loader:
            x = batch["source"]
            mask = batch["masks"]
            metadata = batch["metadata_index"]
            x_in = torch.cat([x, mask], dim=1).to(torch.float).to(device)
            recon, z, mu, logvar = model.vae(x_in)
            latents.append(mu.cpu())
            logvars.append(logvar.cpu())
            recons.append(recon.cpu())
            metadata_list.append(np.array(metadata))

    mus = torch.cat(latents, dim=0)
    logvars = torch.cat(logvars, dim=0)
    recons = torch.cat(recons, dim=0)
    metadata_list = np.concatenate(metadata_list)

    return mus, logvars, recons, metadata_list


def _show(ax, img, title):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LitVAE.load_from_checkpoint(CHECKPOINT, map_location=device, weights_only=False).eval().to(device)

# %%
ds = VideoFrameDataset(
    zarr_path=ZARR_PATH,
    annotations_file=METADATA_PATH_VIDEOS,
    input_array_name="max_projection",
    input_mask_name="masks",
    channels=[0, 1],
)

ds[0]
# %%
ds[0]["source"].shape, ds[0]["target"].shape, ds[0]["masks"].shape
# %%
video_module = BCDataModule(
    ds, batch_size=16, num_workers=7, spatial_transforms=no_transform, intensity_transforms=no_transform
)

video_module.setup("predict")
video_loader = video_module.predict_dataloader()

# %%
mus_video, logvars_video, recons_video, metadata_video = get_latent_features(model, tqdm(video_loader))
Z_video = mus_video.numpy()
df_embed_video = pd.DataFrame(Z_video, columns=[f"z_{i}" for i in range(Z_video.shape[1])])
df_embed_video["image_id"] = metadata_video
meta_video = pd.read_csv(METADATA_PATH_VIDEOS).reset_index(drop=True)[["image_id", "sample_name", "frame_index"]]
df_plot_video = df_embed_video.merge(meta_video, on="image_id")
df_plot_video

# %%
dataset_single = BorderCellDataset(
    annotations_file="/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv",
    zarr_path="/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr",
    input_array_name="max_projection",
    input_mask_name="3D_mask_corrected",
    normalization_function=percentile_norm,
)

single_module = BCDataModule(
    dataset_single, batch_size=32, num_workers=7, spatial_transforms=no_transform, intensity_transforms=no_transform
)

single_module.setup("predict")
single_loader = single_module.predict_dataloader()

mus_single, logvars_single, recons_single, metadata_list_single = get_latent_features(model, tqdm(single_loader))
# %%
Z_single = mus_single.numpy()
df_embed_single = pd.DataFrame(Z_single, columns=[f"z_{i}" for i in range(Z_single.shape[1])])
df_embed_single["image_id"] = metadata_list_single
meta_single = pd.read_csv(METADATA_PATH).reset_index(drop=True)
df_plot_single = df_embed_single.merge(meta_single, on="image_id")
df_plot_single
# %%
df_combined = pd.concat([df_plot_video, df_plot_single], ignore_index=True)
df_combined
# %%
z_cols = [c for c in df_combined.columns if c.startswith("z_")]
Z_all = df_combined[z_cols].values

reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
Z_umap = reducer.fit_transform(Z_all)
df_combined["UMAP1"] = Z_umap[:, 0]
df_combined["UMAP2"] = Z_umap[:, 1]
# %%
is_video = df_combined["frame_index"].notna()
df_vid = df_combined[is_video].copy()
df_vid["frame_index"] = df_vid["frame_index"].astype(int)

videos = df_vid["sample_name"].unique()
fig, axes = plt.subplots(1, len(videos), figsize=(9 * len(videos), 8))
if len(videos) == 1:
    axes = [axes]

for ax, (video_name, traj) in zip(axes, df_vid.groupby("sample_name")):
    ax.scatter(df_combined["UMAP1"], df_combined["UMAP2"], c="gray", s=70, alpha=0.4, linewidths=0, zorder=1)

    traj = traj.sort_values("frame_index")
    x = traj["UMAP1"].values
    y = traj["UMAP2"].values
    t = traj["frame_index"].values

    t_min, t_max = t.min(), t.max()
    t_norm = (t - t_min) / max(t_max - t_min, 1)

    pts = np.stack([x, y], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP, norm=mcolors.Normalize(0, 1), linewidth=1.5, zorder=3)
    lc.set_array(t_norm[:-1])
    ax.add_collection(lc)

    ax.scatter(x, y, c=t_norm, cmap=CMAP, vmin=0, vmax=1, s=20, linewidths=0, zorder=4)

    ax.scatter(x[0], y[0], marker="o", s=80, color=CMAP(0.0), edgecolors="white", linewidths=0.8, zorder=5)
    ax.scatter(x[-1], y[-1], marker="*", s=130, color=CMAP(1.0), edgecolors="white", linewidths=0.8, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(t_min, t_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Frame index")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(video_name)
    ax.autoscale()

fig.suptitle("Latent space — one trajectory per video on full-dataset background", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
id_to_ds_idx = {row["image_id"]: idx for idx, row in ds.metadata.iterrows()}

id_to_recon: dict[str, np.ndarray] = {}
video_dl = tud.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
model.eval()
with torch.no_grad():
    for batch in tqdm(video_dl, desc="video inference"):
        x = batch["source"].to(device)
        mask = batch["masks"].to(device)
        ids = batch["metadata_index"]
        recon_batch, _, _, _ = model.vae(torch.cat([x, mask], dim=1).float())
        for img_id, r in zip(ids, recon_batch.cpu().numpy()):
            id_to_recon[img_id] = r

for video_name, traj in df_vid.groupby("sample_name"):
    traj = traj.sort_values("frame_index").reset_index(drop=True)
    out_dir = OUTPUT_ROOT / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    x_all = traj["UMAP1"].values
    y_all = traj["UMAP2"].values
    t_all = traj["frame_index"].values
    t_min, t_max = t_all.min(), t_all.max()
    t_norm_all = (t_all - t_min) / max(t_max - t_min, 1)

    for i, row in traj.iterrows():
        img_id = row["image_id"]
        t = int(row["frame_index"])

        inp = ds.inputs[id_to_ds_idx[img_id]]
        recon_frame = id_to_recon[img_id]
        n_ch = inp.shape[0]

        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.25)
        ax_umap = fig.add_subplot(gs[0])
        gs_imgs = gs[1].subgridspec(2, n_ch, hspace=0.15, wspace=0.05)

        ax_umap.scatter(df_combined["UMAP1"], df_combined["UMAP2"], c="gray", s=50, alpha=0.4, linewidths=0, zorder=1)

        x_sf = x_all[: i + 1]
        y_sf = y_all[: i + 1]
        tn_sf = t_norm_all[: i + 1]

        if len(x_sf) > 1:
            pts = np.stack([x_sf, y_sf], axis=1)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=CMAP, norm=mcolors.Normalize(0, 1), linewidth=2, zorder=3)
            lc.set_array(tn_sf[:-1])
            ax_umap.add_collection(lc)
            ax_umap.scatter(x_sf[:-1], y_sf[:-1], c=tn_sf[:-1], cmap=CMAP, vmin=0, vmax=1, s=15, linewidths=0, zorder=4)

        ax_umap.scatter(
            x_sf[-1],
            y_sf[-1],
            c=[tn_sf[-1]],
            cmap=CMAP,
            vmin=0,
            vmax=1,
            s=150,
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )

        ax_umap.set_xlabel("UMAP1")
        ax_umap.set_ylabel("UMAP2")
        ax_umap.set_title(f"{video_name}  —  frame {t}", fontsize=9)
        ax_umap.autoscale()

        for ch in range(n_ch):
            _show(fig.add_subplot(gs_imgs[0, ch]), inp[ch], f"input ch{ch}")
            _show(fig.add_subplot(gs_imgs[1, ch]), recon_frame[ch], f"recon ch{ch}")

        fig.savefig(out_dir / f"frame_{t:04d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"{video_name}: {len(traj)} frames saved → {out_dir}")
# %%
for video_name in df_vid["sample_name"].unique():
    out_dir = OUTPUT_ROOT / video_name
    frames = sorted(out_dir.glob("frame_*.png"))

    imgs = [Image.open(f) for f in frames]
    gif_path = OUTPUT_ROOT / f"{video_name}.gif"
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=150,
        loop=0,
    )
    print(f"{video_name}: GIF saved → {gif_path}")
# %%
