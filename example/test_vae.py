#%%
import torchview
from torchview import draw_graph
import torch
from darth_vaeder.models.vae import ResNet18Enc,ResNet18Dec, VAEResNet18
import graphviz

model = ResNet18Dec(nc=2, z_dim=10)
ex_tensor=torch.randn(1, 10)
#%%

graphviz.set_jupyter_format('png')

torchview.draw_graph(model, input_size=ex_tensor.shape)

model_graph_1 = draw_graph(
    model, input_size=ex_tensor.shape,
    depth=3,
    graph_name='VAEResNet18',
    hide_inner_tensors=False,
    hide_module_functions=False,
)
model_graph_1.visual_graph

# %%

#%%
model(ex_tensor)
# %%
