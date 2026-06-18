#%%
import numpy as np

#%%


latents = np.load("/mnt/efs/dl_jrc/student_data/S-EG/project/ShapeEmbedLite/results_32Points/output_BorderCellDM_ls16_e75_b0.05_lr0.001_idx_loss_rfl_loss_cir_pad/test_latent_space.npy")
labels = np.load("/mnt/efs/dl_jrc/student_data/S-EG/project/ShapeEmbedLite/results_32Points/output_BorderCellDM_ls16_e75_b0.05_lr0.001_idx_loss_rfl_loss_cir_pad/test_labels.npy")


# %%
labels

# %%


model_path = "/mnt/efs/dl_jrc/student_data/S-EG/project/ShapeEmbedLite/results_32Points/output_BorderCellDM_ls16_e75_b0.05_lr0.001_idx_loss_rfl_loss_cir_pad/BorderCellDM_model_state_dict.pth"