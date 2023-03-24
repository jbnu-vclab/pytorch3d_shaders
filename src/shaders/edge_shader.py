import torch
import torch.nn.functional as F

from pytorch3d.renderer.mesh.shader import ShaderBase

class EdgeShader(ShaderBase):

    def forward(self, fragments, meshes, **kwargs):
        cameras = super()._get_cameras(**kwargs)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = zfar # (bxWxHx1)

        # normalize depth
        zbuf /= zfar

        # Depth 값으로부터 2D Laplace 필터로 Edge 계산
        zbuf_perm = zbuf.permute(0,3,1,2) # (bx1xWxH)

        laplace_2d_filter = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=zbuf.device) # (3x3)
        laplace_2d_filter = laplace_2d_filter.unsqueeze(0).unsqueeze(0) # (1x1x3x3) # TODO: Batch 고려 필요

        filtered_zbuf = F.conv2d(zbuf_perm, laplace_2d_filter, padding='same')
        filtered_zbuf = filtered_zbuf.permute(0, 2, 3, 1) # (bxWxHx1)

        edge_img = (filtered_zbuf >= 0.001).float() # (bxWxHx1)

        res = torch.cat((zbuf, zbuf, zbuf, edge_img), dim=3) # (bxWxHx4), 1,2,3채널은 정규화 깊이, 4채널은 edge boolean

        return res