from pytorch3d.renderer.mesh.shader import *
from .edge_shader import SoftEdgeShader, HardEdgeShader
from .panoramic_renderer import NaivePanoramicShader

def get_shader_from_name(shader_name: str,
                         device):
    if shader_name.lower() == 'softphong':
        shader = SoftPhongShader(device)
    if shader_name.lower() == 'hardphong':
        shader = HardPhongShader(device)

    if shader_name.lower() == 'hardedge':
        shader = HardEdgeShader(device)
    if shader_name.lower() == 'softedge':
        shader = SoftEdgeShader(device)

    if shader_name.lower() == 'hardpanoramic':
        shader = NaivePanoramicShader(device)

    if shader_name.lower() == 'softsilhouette':
        shader = SoftSilhouetteShader(device)

    return shader