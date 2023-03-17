"""

Implementation from https://www.youtube.com/watch?v=ovB0ddFtzzA&ab_channel=mildlyoverfitted

"""

import numpy as np
import timm
import torch
from vit_reproduce import Vision_Transformer

# Helpers
def get_n_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)

model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

custom_config = {
    "img_size": 384,
    "in_chans": 3,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4
}

model_custom = Vision_Transformer(**custom_config)
model_custom.eval()

for (n_0, p_0), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    assert p_0.numel() == p_c.numel()
    print(f"{n_0} | {n_c}")

    p_c.data[:] = p_0.data

    assert_tensors_equal(p_c.data, p_0.data)

inp = torch.rand(1, 3, 384, 384)
res_c = model_custom(inp)
res_0 = model_official(inp)

# Asserts
assert get_n_parameters(model_custom) == get_n_parameters(model_official)
assert_tensors_equal(res_c, res_0)

# Save custom model
torch.save(model_custom, "model.pth")