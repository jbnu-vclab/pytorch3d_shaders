import os

import pytorch3d.renderer.mesh.shader
import torch
import matplotlib.pyplot as plt

from pytorch3d.io import load_obj, load_objs_as_meshes

from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    TexturesVertex,
    PointLights,
    FoVOrthographicCameras,
    look_at_view_transform,
    RasterizationSettings,
MeshRenderer,
MeshRasterizer,
)

from shaders.get_shader import get_shader_from_name

def load_mesh(model_path: str, load_textures: bool, device):
    if load_textures == False:
        # Add default white material (by texture)
        verts, faces_idx, _ = load_obj(model_path, device=device)
        faces = faces_idx.verts_idx

        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=textures
        )
    else:
        mesh = load_objs_as_meshes([model_path], device=device)

    return mesh

def normalize_mesh(mesh):
    verts = mesh.verts_list()[0]
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center) # in place op
    mesh.scale_verts_((1.0 / float(scale))) # in rendered_imgplace op

# TODO: configurable
def set_lights_and_camera(device):
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    R, T = look_at_view_transform(dist=1, elev=0, azim=0)
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.001, zfar=3)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.001, zfar=2)

    return lights, cameras

# TODO: configurable
def set_raster_setting():
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    return raster_settings


# TODO: configurable
def set_renderer(cameras, lights, raster_settings, shader_name, device):
    shader = get_shader_from_name(shader_name, device)
    shader.device = device
    shader.cameras = cameras
    shader.lights = lights

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    return renderer

def test_shader(model_path: str,
                load_textures: bool,
                normalize: bool,
                shader_name: str,
                device):
    mesh = load_mesh(model_path, load_textures=load_textures, device=device)

    if normalize == True:
        normalize_mesh(mesh)

    lights, cameras = set_lights_and_camera(device)
    raster_settings = set_raster_setting()
    renderer = set_renderer(cameras, lights, raster_settings, shader_name, device)

    rendered_img = renderer(mesh, cameras=cameras, lights=lights)
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_img[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    DATA_DIR = "../data"
    FILENAME = "dolphin.obj"
    model_path = os.path.join(DATA_DIR, FILENAME)

    test_shader(model_path, load_textures=False, normalize=True,
                shader_name="SoftPhong", device=device)

    test_shader(model_path, load_textures=False, normalize=True,
                shader_name="Edge", device=device)