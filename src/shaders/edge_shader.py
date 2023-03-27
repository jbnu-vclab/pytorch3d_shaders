import torch
import torch.nn.functional as F

from pytorch3d.renderer.mesh.shader import ShaderBase

class HardEdgeShader(ShaderBase):

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

class SoftEdgeShader(ShaderBase):

    def forward(self, fragments, meshes, **kwargs):
        if fragments.dists is None:
            raise ValueError("SoftDepthShader requires Fragments.dists to be present.")

        cameras = super()._get_cameras(**kwargs)

        N, H, W, K = fragments.pix_to_face.shape
        device = fragments.zbuf.device
        mask = fragments.pix_to_face >= 0

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))

        # Sigmoid probability map based on the distance of the pixel to the face.
        prob_map = torch.sigmoid(-fragments.dists / self.blend_params.sigma) * mask

        # append extra face for zfar
        dists = torch.cat(
            (fragments.zbuf, torch.ones((N, H, W, 1), device=device) * zfar), dim=3
        )
        probs = torch.cat((prob_map, torch.ones((N, H, W, 1), device=device)), dim=3)

        # compute weighting based off of probabilities using cumsum
        probs = probs.cumsum(dim=3)
        probs = probs.clamp(max=1)
        probs = probs.diff(dim=3, prepend=torch.zeros((N, H, W, 1), device=device))

        zbuf = (probs * dists).sum(dim=3).unsqueeze(3)

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