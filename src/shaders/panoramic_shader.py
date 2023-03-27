import torch
import torch.nn.functional as F

from pytorch3d.renderer.mesh.shader import ShaderBase, HardPhongShader, SoftPhongShader, phong_shading, hard_rgb_blend
from pytorch3d.renderer.cameras import FoVOrthographicCameras, FoVPerspectiveCameras

class HardPanoramicShader(HardPhongShader):
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


