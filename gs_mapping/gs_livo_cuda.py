"""直接编译、调用 GS-LIVO 固定版本的 CUDA forward/backward；不复制上游源码。"""
from pathlib import Path
from functools import lru_cache
import os
import subprocess
import numpy as np
from .render import OPACITY

PIN = "cc65db279bf0d7c4df12d21710cc4053753dd133"


@lru_cache(None)
def extension():
    import torch
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch and an NVIDIA GPU are required")
    if "GS_LIVO_ROOT" not in os.environ:
        raise RuntimeError("Set GS_LIVO_ROOT to the checkout created by scripts/fetch_gs_livo.sh")
    root = Path(os.environ["GS_LIVO_ROOT"]).resolve()
    head = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if head != PIN:
        raise RuntimeError("unexpected GS-LIVO revision: "+head)
    lib = root/"src/lib3dgs"
    cu = lib/"cuda_rasterizer"
    sources = [Path(__file__).parent/"native/bind.cpp",lib/"src/rasterize_points.cu",
               cu/"rasterizer_impl.cu",cu/"forward.cu",cu/"backward.cu"]
    return load(name="lvi_asw_gs_livo_cc65db2",sources=[str(p) for p in sources],
                extra_include_paths=[str(cu),str(lib/"includes"),str(root/"src/glm")],
                extra_cflags=["-O2"],extra_cuda_cflags=["-O2"],verbose=False)


def render(means,covs,colors,k,height,width):
    import torch
    native = extension()
    def tensor(a):
        return torch.as_tensor(a,dtype=torch.float32,device="cuda").contiguous()
    m, c = tensor(means), tensor(colors)
    empty = tensor(np.empty(0))
    packed = tensor(covs[:,(0,0,0,1,1,2),(0,1,2,1,2,2)])
    fx,fy,cx,cy = k
    view = torch.eye(4,device=m.device,dtype=m.dtype)
    # Upstream GLM reads column-major arrays; pixel = ((ndc+1)*size-1)/2.
    proj = tensor([[2*fx/width,0,(2*cx+1)/width-1,0],
                   [0,2*fy/height,(2*cy+1)/height-1,0],
                   [0,0,100/(100-.01),-.01*100/(100-.01)],
                   [0,0,1,0]]).T.contiguous()
    bg = tensor(np.zeros(3))
    op = torch.full((len(m),1),OPACITY,device=m.device,dtype=m.dtype)
    tx, ty = width/(2*fx), height/(2*fy)
    def forward(colors_t):
        return native.forward(bg,m,colors_t,op,empty,empty,1.,packed,view,proj,
                              tx,ty,height,width,empty,0,bg,False,False)
    num, rgb, radii, geom, bins, im = forward(c)
    _, ones, *_ = forward(torch.ones_like(c))
    def vjp(grad):
        gc = tensor(np.moveaxis(grad,-1,0).copy())
        # RasterizerBackward returns dmean2D,dcolor,dopacity,dmean3D,...
        _, dc, *_ = native.backward(bg,m,radii,c,empty,empty,1.,packed,view,proj,
                                     tx,ty,gc,empty,0,bg,geom,num,bins,im,False)
        return dc.detach().cpu().numpy()
    return (rgb.permute(1,2,0).cpu().numpy(), ones[0].cpu().numpy(), vjp)
