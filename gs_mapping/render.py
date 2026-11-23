"""前向合成和对 RGB 的伴随；CPU 是可执行参考，cuda 后端调用原版 GS-LIVO。"""
import numpy as np
from .geometry import pixel

OPACITY = .85


def cpu(means, covs, colors, k, height, width):
    rgb = np.zeros((height, width, 3))
    trans = np.ones((height, width))
    blocks = []
    for i in np.argsort(means[:,2], kind="stable"):
        if means[i,2] <= .2:
            continue
        uv, j = pixel(k, means[i])
        c = j @ covs[i] @ j.T + .3*np.eye(2)
        rad = 3*np.sqrt(np.linalg.eigvalsh(c)[-1])
        x0,y0 = np.maximum(np.floor(uv-rad),0).astype(int)
        x1,y1 = np.minimum(np.ceil(uv+rad)+1,[width,height]).astype(int)
        if x0>=x1 or y0>=y1:
            continue
        sl = np.s_[y0:y1,x0:x1]
        y,x = np.mgrid[y0:y1,x0:x1]
        d = np.stack([x-uv[0], y-uv[1]],axis=-1)
        a = OPACITY*np.exp(-.5*np.einsum('...i,ij,...j->...',d,np.linalg.inv(c),d))
        a[(a<1/255.) | (trans[sl]<1e-4)] = 0.
        weight = trans[sl]*a
        rgb[sl] += weight[:,:,None]*colors[i]
        trans[sl] *= (1-a)
        blocks.append((i,sl,weight))
    def vjp(grad):
        out = np.zeros_like(colors)
        for idx,sl,w in blocks:
            out[idx] = np.einsum('ij,ijc->c',w,grad[sl])
        return out
    return rgb, 1-trans, vjp


def renderer(name):
    if name == "cpu":
        return cpu
    if name == "cuda":
        from .gs_livo_cuda import render
        return render
    raise ValueError(name)


def color_terms(render, means, cov, colors, k, image, valid, min_coverage):
    h, w = image.shape[:2]
    pred, alpha, vjp = render(means,cov,colors,k,h,w)
    mask = valid & (alpha >= min_coverage)
    err = pred-image
    # Huber influence in RGB intensity (linear algebra uses [0,1] sRGB samples).
    grad = np.maximum(-.10,np.minimum(.10,err))*mask[:,:,None]
    denom = vjp(np.repeat(mask[:,:,None],3,axis=2).astype(float))
    return vjp(grad), denom, np.sum(np.abs(err[mask])), int(mask.sum())*3
