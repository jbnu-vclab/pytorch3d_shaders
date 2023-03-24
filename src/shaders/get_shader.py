from pytorch3d.renderer.mesh.shader import *
from .edge_shader import EdgeShader

def get_shader_from_name(shader_name: str,
                         device):
    if shader_name.lower() == 'softphong':
        shader = SoftPhongShader(device)
    if shader_name.lower() == 'edge':
        shader = EdgeShader(device)

    return shader