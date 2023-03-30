import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pytorch3d.renderer.mesh.shader import ShaderBase, HardPhongShader, SoftPhongShader, phong_shading, hard_rgb_blend
from pytorch3d.renderer.cameras import FoVOrthographicCameras, FoVPerspectiveCameras
from pytorch3d.renderer import MeshRenderer, MeshRasterizer
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import RotateAxisAngle

# (deprecated)
class NaivePanoramicShader(HardPhongShader):
    # TODO: (Important!!) 아래 render_side에 구현된 것처럼 렌더링 시점에 정점에 transformation을 적용하는 것은 현재 불가능함
    # phong shading에서는 정점이 이미 변환되었다고 가정하고 프로세스가 구현되어 있음
    # 따라서 하나의 셰이더안의 forward 과정에서 아래와 같은 방식으로 구현하는 것은 불가능하며, 큐브맵 생성을 하려면 test_shader.py의
    # set_renderer를 하는 시점에 서로 다른 자세를 가진 카메라들로 여러 이미지를 생성 후에 합치는 과정이 필요함


    def forward(self, fragments, meshes, **kwargs):
        cameras = super()._get_cameras(**kwargs)
        self.texels = meshes.sample_textures(fragments)
        self.lights = kwargs.get("lights", self.lights)
        self.materials = kwargs.get("materials", self.materials)
        self.blend_params = kwargs.get("blend_params", self.blend_params)

        R_f, T = cameras.R, cameras.T # base forward direction
        R_b = torch.cat((-R_f[:,:,0], R_f[:,:,1], -R_f[:,:,2]),dim=0).unsqueeze(0)
        R_l = torch.cat((R_f[:,:,2], R_f[:,:,1], -R_f[:,:,0]),dim=0).unsqueeze(0)
        R_r = torch.cat((-R_f[:, :, 2], R_f[:, :, 1], R_f[:, :, 0]),dim=0).unsqueeze(0)
        R_t = torch.cat((R_f[:, :, 0], -R_f[:, :, 2], R_f[:, :, 1]),dim=0).unsqueeze(0)
        R_bot = torch.cat((R_f[:, :, 0], R_f[:, :, 2], -R_f[:, :, 1]), dim=0).unsqueeze(0)

        zfar = cameras.zfar
        znear = cameras.znear

        img_f = self.render_side(R_f, T, zfar, znear, cameras.device)
        img_b = self.render_side(R_b, T, zfar, znear, cameras.device)
        img_l = self.render_side(R_l, T, zfar, znear, cameras.device)
        img_r = self.render_side(R_r, T, zfar, znear, cameras.device)
        img_t = self.render_side(R_t, T, zfar, znear, cameras.device)
        img_bot = self.render_side(R_bot, T, zfar, znear, cameras.device)

        # TODO: temporary img output for testing
        img = torch.cat((img_f,img_b,img_l,img_r,img_t,img_bot), dim=1)

        return img

    def render_side(self, R, T, meshes, fragments, zfar, znear, device):
        cam = FoVPerspectiveCameras(device=device, fov=90., aspect_ratio=1., R=R, T=T, zfar=zfar, znear=znear)
        colors = phong_shading(
            meshes=self.meshes,
            fragments=self.fragments,
            texels=self.texels,
            lights=self.lights,
            cameras=cam,
            materials=self.materials,
        )
        images = hard_rgb_blend(colors, self.fragments, self.blend_params)

        return images


class PanoramicRendering(nn.Module):
    def __init__(self,
                 init_pos,
                 init_rot,
                 pano_img_width,
                 pano_img_height,
                 mesh,
                 lights,
                 raster_settings,
                 base_shader,
                 blend_params,
                 image_ref,
                 device):
        super(PanoramicRendering, self).__init__()
        self.pos = nn.Parameter(init_pos.to(device))
        self.rot = nn.Parameter(init_rot.to(device))

        self.cube_width = raster_settings.image_size
        self.pano_img_width = pano_img_width
        self.pano_img_height = pano_img_height

        self.base_shader = base_shader
        self.device = device
        self.lights = lights
        self.raster_settings = raster_settings
        self.blend_params = blend_params

        self.meshes = mesh.extend(6).clone() # TODO: mesh should be properly "batched" to match cubemap (6 dir)

        rad_90 = torch.tensor([90.* torch.pi / 180.])
        y_axis = torch.tensor([0,1,0])
        x_axis = torch.tensor([1,0,0])

        # (Important Note!!) camera T & R is applied to transform the scene to view space.
        # It does not describe pose of the camera
        self.rotation_matrices = torch.stack((self.rotation_matrix(y_axis, 0 * rad_90),  # forward
                                              self.rotation_matrix(y_axis, -rad_90),    # left
                                              self.rotation_matrix(y_axis, 2 * rad_90), # back
                                              self.rotation_matrix(y_axis, rad_90),   # right
                                              self.rotation_matrix(x_axis, rad_90),    # up
                                              self.rotation_matrix(x_axis, -rad_90),   # down
                                              ), dim=0).to(device)

        self.cubemap_to_pano_mapping_flat = self.get_projection_mapping_table()
        self.cubemap_to_pano_mapping_flat.to(device)

        if image_ref is not None:
            buf_img = torch.from_numpy((image_ref[...,:3].max(-1) != 1).astype(np.float32))
            self.register_buffer('image_ref', buf_img)

        pass

    def set_shader(self, cameras):
        self.base_shader.device = self.device
        self.base_shader.cameras = cameras
        self.base_shader.lights = self.lights
        if self.blend_params:
            self.base_shader.blend_params = self.blend_params

    def set_renderer(self, cameras):
        self.set_shader(cameras)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=self.base_shader
        )

        return renderer

    # https://gist.github.com/fgolemo/94b5caf0e209a6e71ab0ce2d75ad3ed8
    def rotation_matrix(self, axis, theta):
        """
        Generalized 3d rotation via Euler-Rodriguez formula, https://www.wikiwand.com/en/Euler%E2%80%93Rodrigues_formula
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = axis / torch.sqrt(torch.dot(axis, axis))
        a = torch.cos(theta / 2.0)
        b, c, d = -axis * torch.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

        col_major_rot = torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                             [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                             [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        return torch.transpose(col_major_rot,1,0)

    def debug_draw(self, tensor, is_gray = False):
        import matplotlib.pyplot as plt

        img = tensor.detach().cpu().numpy()

        plt.figure(figsize=(10, 5))
        if is_gray == False:
            plt.imshow(img[...,:])
        else:
            plt.imshow(img[..., 0])

        plt.show()

    def get_projection_mapping_table(self):
        DEG_TO_RAD = torch.pi / 180.

        x = torch.arange(-180.,180., 360./self.pano_img_width).flip(dims=[0])
        y = torch.arange(-90., 90., 180. / self.pano_img_height)
        grid_x, grid_y = torch.meshgrid(x,y)

        grid_x = grid_x * DEG_TO_RAD
        grid_y = grid_y * DEG_TO_RAD

        # https://stackoverflow.com/questions/34250742/converting-a-cubemap-into-equirectangular-panorama

        # modified some calculation becuase we want z-axis is looking dir
        pixel_vector = torch.stack((torch.cos(grid_y) * torch.sin(grid_x),
                                    -torch.sin(grid_y),
                                    torch.cos(grid_y) * torch.cos(grid_x),
                                    ),
                                   dim=0)

        val, _ = torch.max(torch.abs(pixel_vector), dim=0)
        projected_vec = torch.div(pixel_vector, val)

        # self.debug_draw(((projected_vec + 1.)/2.).permute(1,2,0))

        offset = torch.zeros(2,self.pano_img_width,self.pano_img_height, dtype=torch.long)

        front_x_off = (((projected_vec[0, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        front_y_off = (((projected_vec[1, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        left_x_off = (((projected_vec[2, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        left_y_off = (((projected_vec[1, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        right_x_off = (((-projected_vec[2, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        right_y_off = (((projected_vec[1, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        back_x_off = (((-projected_vec[0, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        back_y_off = (((projected_vec[1, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        up_x_off = (((projected_vec[0, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        up_y_off = (((-projected_vec[2, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        down_x_off = (((projected_vec[0, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)
        down_y_off = (((projected_vec[2, :, :] + 1) / 2) * self.cube_width).type(torch.LongTensor)

        offset[0, :, :] += torch.where(projected_vec[0, : ,:] == 1, right_x_off, 0)
        offset[1, :, :] += torch.where(projected_vec[0, :, :] == 1, right_y_off, 0)
        offset[0, :, :] += torch.where(projected_vec[0, :, :] == -1, left_x_off, 0)
        offset[1, :, :] += torch.where(projected_vec[0, :, :] == -1, left_y_off, 0)
        offset[0, :, :] += torch.where(projected_vec[1, :, :] == 1, up_x_off, 0)
        offset[1, :, :] += torch.where(projected_vec[1, :, :] == 1, up_y_off, 0)
        offset[0, :, :] += torch.where(projected_vec[1, :, :] == -1, down_x_off, 0)
        offset[1, :, :] += torch.where(projected_vec[1, :, :] == -1, down_y_off, 0)
        offset[0, :, :] += torch.where(projected_vec[2, :, :] == 1, front_x_off, 0)
        offset[1, :, :] += torch.where(projected_vec[2, :, :] == 1, front_y_off, 0)
        offset[0, :, :] += torch.where(projected_vec[2, :, :] == -1, back_x_off, 0)
        offset[1, :, :] += torch.where(projected_vec[2, :, :] == -1, back_y_off, 0)

        # truncate result
        offset = torch.clamp(offset,0,self.cube_width - 1)

        # self.debug_draw((offset[0,:,:]/self.cube_width).unsqueeze(-1), is_gray=True)
        # self.debug_draw((offset[1, :, :] / self.cube_width).unsqueeze(-1), is_gray=True)

        # image index (of cubemap_imgs): f(0), l(1), b(2), r(3), u(4), d(5)
        texture_ind = torch.zeros(1,self.pano_img_width,self.pano_img_height,dtype=torch.long)
        texture_ind[0, :, :] += torch.where(projected_vec[0, :, :] == 1, 3, 0)
        texture_ind[0, :, :] += torch.where(projected_vec[0, :, :] == -1, 1, 0)
        texture_ind[0, :, :] += torch.where(projected_vec[1, :, :] == -1, 4, 0)
        texture_ind[0, :, :] += torch.where(projected_vec[1, :, :] == 1, 5, 0)
        texture_ind[0, :, :] += torch.where(projected_vec[2, :, :] == 1, 0, 0)
        texture_ind[0, :, :] += torch.where(projected_vec[2, :, :] == -1, 2, 0)

        # self.debug_draw(((texture_ind.type(torch.FloatTensor) + 1.)/6.).permute(1,2,0), is_gray=True)

        cubemap_to_pano_mapping = (texture_ind * self.cube_width * self.cube_width) + \
                                  (offset[1,:,:] * self.cube_width) + offset[0,:,:]

        # truncate result (why?)
        cubemap_to_pano_mapping = torch.clamp(cubemap_to_pano_mapping,0,6 * self.cube_width * self.cube_width - 1)

        cubemap_to_pano_mapping = cubemap_to_pano_mapping.transpose(1,0)

        cubemap_to_pano_mapping_flat = cubemap_to_pano_mapping.flatten()

        return cubemap_to_pano_mapping_flat

    # TODO: validate for batch operation
    def render_panorama(self):
        # (Important Note!!) camera T & R is applied to transform the scene to view space.
        # It does not describe pose of the camera

        poss = self.pos.repeat(6, 1)
        rots = torch.matmul(self.rot.unsqueeze(0), self.rotation_matrices)

        cameras = FoVPerspectiveCameras(device=self.device, R=rots, T=poss, znear=0.001, zfar=10,
                                        fov=90.0, aspect_ratio=1.0)
        renderer = self.set_renderer(cameras)
        cubemap_imgs = renderer(self.meshes, cameras=cameras, lights=self.lights)  # (bxWxHx4) # Mesh does not propagate grad

        # cubemap concat to debugging
        cubemap_imgs_cat = torch.cat((cubemap_imgs[0, :, :, :], cubemap_imgs[1, :, :, :],
                                      cubemap_imgs[2, :, :, :], cubemap_imgs[3, :, :, :],
                                      cubemap_imgs[4, :, :, :], cubemap_imgs[5, :, :, :]),
                                     dim=1)

        cubemap_imgs_flatten = cubemap_imgs.view([6 * self.cube_width * self.cube_width, 4])

        equirect_img = cubemap_imgs_flatten[self.cubemap_to_pano_mapping_flat, :]

        equirect_img = equirect_img.view([self.pano_img_width, self.pano_img_height, 4]).permute(1, 0, 2).flip([0, 1])

        equirect_img = equirect_img.unsqueeze(0)
        cubemap_imgs_cat = cubemap_imgs_cat.unsqueeze(0)

        return equirect_img, cubemap_imgs_cat

    def forward(self):
        equirect_img, _ = self.render_panorama()

        if self.training:
            loss = torch.sum((equirect_img[...,3] - self.image_ref) ** 2)
            return equirect_img, loss
        else:
            return equirect_img, torch.tensor([0])


