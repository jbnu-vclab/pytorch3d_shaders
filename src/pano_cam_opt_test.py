import os

import pytorch3d.transforms
import torch
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from skimage import img_as_ubyte

from pytorch3d.io import load_obj, load_objs_as_meshes

from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene

from pytorch3d.renderer import (
    TexturesVertex,
    PointLights,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
    look_at_rotation,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
)

from shaders.get_shader import get_shader_from_name

def load_mesh(model_path, load_textures, device):
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
    mesh.scale_verts_((1.0 / float(scale))) # in-place op

# TODO: configurable
def set_lights_and_camera(device, is_cubemap):
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    if is_cubemap == False:
        # R, T = look_at_view_transform(dist=1, elev=0, azim=180)
        R, T = look_at_view_transform(dist=1, eye=((0.0,0.0,0.0),), at=((0.0,0.0,1.0),))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.001, zfar=10)
        # cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.001, zfar=2)
    else:
        R, T = look_at_view_transform(dist=1, eye=((0.0, 0.0, 0.0),),
                                      at=((0.0, 0.0, 1.0),
                                          (0.0, 0.0, -1.0),
                                          (1.0, 0.0, 0.0),
                                          (-1.0, 0.0, 0.0),
                                          (0.0, 1.0, 1e-10),
                                          (0.0, -1.0, 1e-10),))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.001, zfar=10,
                                        fov=90.0, aspect_ratio=1.0)

    return lights, cameras

# TODO: configurable
def set_raster_setting(blend_params=None):
    import numpy as np
    if blend_params:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
        )
    else:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0,
            faces_per_pixel=1,
        )

    return raster_settings


# TODO: configurable
def set_renderer(cameras, lights, raster_settings, shader_name, blend_params, device):
    shader = get_shader_from_name(shader_name, device)
    shader.device = device
    shader.cameras = cameras
    shader.lights = lights
    shader.blend_params = blend_params

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    return renderer

def pano_cam_opt(model_paths: list,
                model_translations: list,
                load_textures: bool,
                normalize: bool,
                shader_name: str,
                device):
    from src.shaders.panoramic_renderer import PanoramicRendering

    # 1) prepare scene and renderer---

    assert len(model_translations) == len(model_paths)
    mesh_list = [load_mesh(model_path, load_textures=load_textures, device=device) for model_path in model_paths]

    if normalize == True:
        [normalize_mesh(mesh) for mesh in mesh_list]

    [mesh.offset_verts_(torch.tensor(tr, device=device)) for mesh, tr in zip(mesh_list, model_translations)]
    mesh = join_meshes_as_scene(mesh_list)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    lights, _ = set_lights_and_camera(device, is_cubemap=False)
    raster_settings = set_raster_setting(blend_params)
    base_shader = get_shader_from_name("HardPhong", device)

    # 2) create reference image---

    target_pos = torch.Tensor([0,0,0])
    target_rot = torch.Tensor([[1,0,0], [0,1,0], [0,0,1]])

    ref_generater = PanoramicRendering(target_pos,target_rot,512,256,mesh, lights, raster_settings, base_shader, blend_params, None, device)
    ref_generater.eval()
    image_ref, _ = ref_generater()
    image_ref = image_ref.detach().cpu().numpy()

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_ref.squeeze())  # only plot the alpha channel of the RGBA image
    # plt.grid(False)
    # plt.show()

    # 3) Init model and optimizer---

    filename_output = "./pano_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I')

    init_pos = torch.Tensor([0, 0, 0])
    init_rot = pytorch3d.transforms.random_rotation()

    base_shader = get_shader_from_name("SoftSilhouette", device)

    model = PanoramicRendering(init_pos, init_rot, 512, 256, mesh, lights, raster_settings, base_shader, blend_params,
                               image_ref, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    plt.figure(figsize=(10, 10))

    # image_init, _ = model()
    # plt.subplot(1, 2, 1)
    # plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
    # plt.grid(False)
    # plt.title("Starting position")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(model.image_ref.cpu().numpy().squeeze())
    # plt.grid(False)
    # plt.title("Reference silhouette")
    # plt.show()

    # 4) Run optimizer

    loop = tqdm(range(200))
    for i in loop:
        optimizer.zero_grad()
        _, loss = model()
        loss.backward()
        optimizer.step()

        loop.set_description('Optimizing (loss %.4f)' % loss.data)

        if loss.item() < 200:
            break

        print(model.rot.detach().cpu())

        # # Save outputs to create a GIF.
        # if i % 10 == 0:
        #     model.eval()
        #     image, _ = model()
        #     image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        #
        #     plt.imshow(image)
        #
        #     image = img_as_ubyte(image)
        #     writer.append_data(image)
        #
        #     plt.figure()
        #     plt.imshow(image[..., :3])
        #     plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
        #     plt.axis("off")

    writer.close()



if __name__ == '__main__':
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    DATA_DIR = "../data"
    FILENAME = ["dolphin.obj",
                "dolphin.obj",
                "LCD_Monitor.obj",
                "LCD_Monitor.obj",
                "dolphin.obj",
                "dolphin.obj"]
    model_paths = [os.path.join(DATA_DIR, x) for x in FILENAME]

    model_translations = [[0.,1.,3.], [-1.,0.,-3.], [-3., 0.5, 0.], [3., 0.2, 0.], [1., 3., 0.], [2., -7., 0.]]

    pano_cam_opt(model_paths, load_textures=False, model_translations=model_translations, normalize=True,
                     shader_name="SoftSilhouette", device=device)