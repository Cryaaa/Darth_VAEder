"""Extract latent embeddings for all cells from a trained checkpoint.

Writes outputs/embeddings/embeddings.npz with:
  z          (N, z_dim)  latent means
  ncells_idx (N,)        global cell index
  condition  (N,)        condition string per cell
  replicate  (N,)        replicate string per cell
"""
