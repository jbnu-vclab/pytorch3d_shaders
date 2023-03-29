import os

import pytorch3d.renderer.mesh.shader
import torch
import matplotlib.pyplot as plt

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
    BlendParams
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

# TODO: consider batch
def test_pano_render(model_paths: list,
                model_translations: list,
                load_textures: bool,
                normalize: bool,
                shader_name: str,
                device):
    assert len(model_translations) == len(model_paths)
    mesh_list = [load_mesh(model_path, load_textures=load_textures, device=device) for model_path in model_paths]

    if normalize == True:
        [normalize_mesh(mesh) for mesh in mesh_list]

    [mesh.offset_verts_(torch.tensor(tr, device=device)) for mesh, tr in zip(mesh_list, model_translations)]

    mesh = join_meshes_as_scene(mesh_list)
    meshes = mesh.extend(6)  # mesh should be "batched" to match cubemap (6 dir)

    lights, cameras = set_lights_and_camera(device, is_cubemap=True)
    raster_settings = set_raster_setting()
    renderer = set_renderer(cameras, lights, raster_settings, shader_name, device)

    rendered_img = renderer(meshes, cameras=cameras, lights=lights) # (bxWxHx4)

    debug_img = torch.cat((rendered_img[0,:,:],rendered_img[1,:,:],
                           rendered_img[2,:,:],rendered_img[3,:,:],
                           rendered_img[4,:,:],rendered_img[5,:,:]), dim=1) # (bWxHx4)
    debug_img = debug_img.unsqueeze(0)

    plt.figure(figsize=(10, 5))

    plt.subplot(2,1,1)
    plt.title("color channel")
    plt.imshow(debug_img[0, ..., :3].cpu().numpy())
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.title("alpha channel")
    plt.imshow(debug_img[0, ..., 3].cpu().numpy())
    plt.axis("off")

    plt.show()


def test_shader(model_paths: list,
                model_translations: list,
                load_textures: bool,
                normalize: bool,
                shader_name: str,
                device):
    assert len(model_translations) == len(model_paths)
    mesh_list = [load_mesh(model_path, load_textures=load_textures, device=device) for model_path in model_paths]

    if normalize == True:
        [normalize_mesh(mesh) for mesh in mesh_list]

    [mesh.offset_verts_(torch.tensor(tr, device=device)) for mesh, tr in zip(mesh_list, model_translations)]

    mesh = join_meshes_as_scene(mesh_list)

    lights, cameras = set_lights_and_camera(device, is_cubemap=False)
    raster_settings = set_raster_setting()
    renderer = set_renderer(cameras, lights, raster_settings, shader_name, device)

    rendered_img = renderer(mesh, cameras=cameras, lights=lights) # (bxWxHx4)

    plt.figure(figsize=(10, 20))

    plt.subplot(1,2,1)
    plt.title("color channel")
    plt.imshow(rendered_img[0, ..., :3].cpu().numpy())
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("alpha channel")
    plt.imshow(rendered_img[0, ..., 3].cpu().numpy())
    plt.axis("off")

    plt.show()

def test_pano_module(model_paths: list,
                model_translations: list,
                load_textures: bool,
                normalize: bool,
                shader_name: str,
                device):
    from src.shaders.panoramic_renderer import PanoramicRendering

    assert len(model_translations) == len(model_paths)
    mesh_list = [load_mesh(model_path, load_textures=load_textures, device=device) for model_path in model_paths]

    if normalize == True:
        [normalize_mesh(mesh) for mesh in mesh_list]

    [mesh.offset_verts_(torch.tensor(tr, device=device)) for mesh, tr in zip(mesh_list, model_translations)]
    mesh = join_meshes_as_scene(mesh_list)

    blend_params = None

    lights, _ = set_lights_and_camera(device, is_cubemap=False)
    raster_settings = set_raster_setting()
    base_shader = get_shader_from_name(shader_name, device)


    init_pos = torch.Tensor([0,0,0])
    # init_rot = torch.Tensor([[1,0,0], [0,1,0], [0,0,1]])
    init_rot = pytorch3d.transforms.random_rotation()

    model = PanoramicRendering(init_pos,init_rot,1024,512,mesh, lights, raster_settings, base_shader, blend_params, device)

    model.eval()
    rendered_img, cubemap_imgs_cat = model()
    rendered_img, cubemap_imgs_cat = rendered_img.detach(), cubemap_imgs_cat.detach()

    for i in range(6):
        cubemap_imgs_cat[:,:,i*raster_settings.image_size-1,:] = 0

    plt.figure(figsize=(10, 5))

    plt.subplot(2,1,1)
    plt.title("pono img")
    plt.imshow(rendered_img[0, ..., :3].cpu().numpy())

    plt.subplot(2, 1, 2)
    plt.title("cubemap img")
    plt.imshow(cubemap_imgs_cat[0, ..., :3].cpu().numpy())

    plt.show()



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
                "dolphin.obj",
                "LCD_Monitor.obj",
                "dolphin.obj",
                "dolphin.obj"]
    model_paths = [os.path.join(DATA_DIR, x) for x in FILENAME]

    model_translations = [[0.,1.,3.], [-1.,0.,-3.], [-3., 0.5, 0.], [3., 0.2, 0.], [1., 3., 0.], [2., -7., 0.]]

    # test_shader(model_paths, load_textures=False, model_translations=model_translations, normalize=True,
    #             shader_name="SoftSilhouette", device=device)

    # test_shader(model_paths, load_textures=False, model_translations=model_translations, normalize=True,
    #             shader_name="SoftPhong", device=device)

    # test_shader(model_path, load_textures=False, normalize=True,
    #             shader_name="HardEdge", device=device)
    #
    # test_shader(model_path, load_textures=False, normalize=True,
    #             shader_name="SoftEdge", device=device)

    # test_shader(model_paths, load_textures=False, model_translations=model_translations, normalize=True,
    #             shader_name="HardPanoramic", device=device)

    # test_pano_render(model_paths, load_textures=False, model_translations=model_translations, normalize=True,
    #              shader_name="SoftPhong", device=device)

    test_pano_module(model_paths, load_textures=False, model_translations=model_translations, normalize=True,
                     shader_name="SoftPhong", device=device)