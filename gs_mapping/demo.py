"""Synthetic oblique views of colored LiDAR surfaces; no measured dataset results."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import apply, inv, transform, shape
from .render import cpu


def generate(path):
    path=Path(path); path.mkdir(parents=True,exist_ok=False)
    k=[46.,47.,22.,14.]  # deliberately off-centre
    xyz=np.array([[x,y,2.5+.35*(x>0)] for y in (-.46,-.2,.07,.36) for x in (-.75,-.46,-.18,.12,.43,.73)])
    colors=np.array([[.25+.3*(i%3),.2+.1*(i%5),.7-.16*(i%4)] for i in range(len(xyz))])
    normals=np.tile([0.,0.,-1.],(len(xyz),1))
    cov=np.stack([shape(n,.095,.025) for n in normals])
    extr=[.035,0.,.01,0,0,0,1]
    def pose(t):
        return [t*.35,0.,0.,0.,0.,0.,1.]
    def knot(i,t,active=True):
        return dict(id=i+1,t=t,T=pose(t),P=(.0001*np.eye(6)).ravel().tolist(),active=active)
    events=[dict(kind="header",version=1)]
    for i in range(9):
        t=round(.12*i,6)
        scan=apply(inv(transform(pose(t))),xyz)
        rows=np.column_stack((scan,normals,np.full(len(xyz),.0004)))
        events.append(dict(kind="scan",t=t,rows=rows.tolist()))
        # Some earlier poses are explicitly in their final pre-removal solve.
        events.append(dict(kind="solve",T_CI=extr,poses=[knot(j,round(.12*j,6),j>i-4)
                                                            for j in range(max(0,i-4),i+1)]))
        if i==8:
            break
        ti=round(t+.035,6)
        twc=transform(extr)@inv(transform(pose(ti)))
        rgb,_,_=cpu(apply(twc,xyz),cov,colors,k,32,48)
        filename="rgb_%02d.png"%i
        Image.fromarray(np.rint(255*rgb).astype("uint8")).save(path/filename)
        events.append(dict(kind="image",t=ti,file=filename,K=k))
    events.append(dict(kind="end"))
    (path/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    return events


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    generate(p.parse_args().directory)
