"""Ordered journal reader, independent image/scan association, map export."""
from collections import deque, Counter
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .geometry import transform, apply
from .mapper import Mapper, Frame
from .render import OPACITY


def records(path, follow=False):
    with Path(path).open(encoding="utf8") as stream:
        while True:
            pos=stream.tell()
            line=stream.readline()
            if line.endswith("\n"):
                obj=json.loads(line)
                yield obj
                if obj["kind"]=="end":
                    return
            elif not follow:
                if line:
                    raise ValueError("truncated journal line")
                return
            else:
                stream.seek(pos)
                time.sleep(.05)


def asset(root, name):
    if Path(name).name!=name or not name:
        raise ValueError("payload must be a basename")
    return root/name


def pixels(root, e):
    with Image.open(asset(root,e["file"])) as f:
        img=np.asarray(f.convert("RGB"),dtype=float)/255.
    if "mask" in e:
        with Image.open(asset(root,e["mask"])) as f:
            mask=np.asarray(f.convert("L"))>0
    else:
        mask=np.ones(img.shape[:2],bool)
    k=np.asarray(e["K"],float).reshape(4)
    if mask.shape!=img.shape[:2] or not np.isfinite(k).all() or min(k[:2])<=0:
        raise ValueError("invalid rectified image")
    return img,mask,k


def write_ply(path, m):
    xyz,ns,covs=m.world()
    props=["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity",
           "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    with Path(path).open("w",encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n"%len(xyz))
        for p in props:
            f.write("property float "+p+"\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p,n,cov,g in zip(xyz,ns,covs,m.map):
            vals,axes=np.linalg.eigh(cov)
            axes=axes[:,::-1].copy()
            if np.linalg.det(axes)<0:
                axes[:,0]*=-1
            logscales=.5*np.log(vals[::-1])
            quat=Rotation.from_matrix(axes).as_quat()
            values=[*p,*n,*((g.color-.5)/.28209479177387814),np.log(OPACITY/(1-OPACITY)),
                    *logscales,*quat[[3,0,1,2]]]
            f.write(" ".join("%.9g"%v for v in values)+" "+
                    " ".join(str(int(v)) for v in np.rint(255*g.color))+"\n")


class Session:
    def __init__(self,root,out,cfg,backend,holdout=0):
        self.root,self.out,self.cfg,self.backend=Path(root),Path(out),cfg,backend
        self.holdout=holdout
        self.epoch=0
        self.header=False
        self.reports=[]
        self._new()

    def _new(self):
        self.mapper=Mapper(self.cfg,self.backend)
        self.scans,self.images=deque(),deque()
        self.last_t={"scan":-np.inf,"image":-np.inf}
        self.count=Counter()
        self.held=[]
        self.log=(self.out/("window_%03d.jsonl"%self.epoch)).open("w",encoding="utf8")

    def event(self,e):
        kind=e["kind"]
        if kind=="header":
            if self.header or e.get("version")!=1:
                raise ValueError("duplicate or unsupported journal header")
            self.header=True
            return
        if not self.header:
            raise ValueError("missing journal header")
        if kind=="reset":
            if int(e["epoch"])<=self.epoch:
                raise ValueError("non-increasing reset epoch")
            self.flush()
            self.epoch=int(e["epoch"])
            self._new()
        elif kind in ("scan","image"):
            t=float(e["t"])
            if not np.isfinite(t):
                raise ValueError("nonfinite observation time")
            if t<=self.last_t[kind]:
                self.count["duplicate_or_late_"+kind]+=1
                return
            self.last_t[kind]=t
            self.count[kind]+=1
            if kind=="image":
                self.images.append((e,self.holdout>0 and self.count[kind]%self.holdout==0))
            else:
                rows=np.asarray(e["rows"],float).reshape(-1,7)
                rows=rows[np.isfinite(rows).all(axis=1)&(rows[:,6]>0)]
                norm=np.linalg.norm(rows[:,3:6],axis=1)
                rows=rows[norm>1e-6].copy()
                rows[:,3:6]/=norm[norm>1e-6,None]
                self.scans.append((t,rows))
            # With no state solutions/sensor dropouts, keep input buffers bounded.
            for q in (self.scans,self.images):
                while len(q)>max(32,8*self.cfg.max_candidates):
                    q.popleft(); self.count["queue_overflow"]+=1
        elif kind=="solve":
            self.mapper.ex=transform(e["T_CI"])
            self.mapper.trajectory.update(e["poses"])
            self.drain()
            self.compact()
        elif kind!="end":
            raise ValueError("unknown event: "+str(kind))

    def compact(self):
        tr=self.mapper.trajectory
        if not tr.knots:
            return
        latest=tr.times[-1]
        cutoff=latest-self.cfg.history-self.cfg.max_gap-2*self.cfg.pair_dt
        refs={g.anchor for g in self.mapper.map}
        for f in self.mapper.frames:
            refs.update((f.ref,f.scan_ref))
        refs.update(ref for _,ref in self.held)
        keep={i for ref in refs for i in (ref.a,ref.b)}
        before=[(t,i) for t,i in tr.ordered if t<cutoff]
        if before:
            keep.add(before[-1][1])
        drop=[i for t,i in before if i not in keep and not tr.knots[i][3]]
        if drop:
            for i in drop:
                del tr.knots[i]
            tr.update([])

    def drain(self,final=False):
        m,c=self.mapper,self.cfg
        tr=m.trajectory
        while self.images:
            e,hold=self.images[0]; t=float(e["t"])
            # Wait until both the upper pose knot and nearest-scan search horizon
            # are known. End-of-stream never licenses pose extrapolation.
            if not final and (not tr.knots or tr.times[-1] < t+c.pair_dt or
                              self.last_t["scan"] < t+c.pair_dt):
                break
            self.images.popleft()
            iref=tr.ref(t,c.max_gap)
            if iref is None:
                self.count["no_image_pose"]+=1; continue
            if hold:
                self.held.append((e,iref)); continue
            while self.scans and self.scans[0][0]<t-c.pair_dt-1e-8:
                self.scans.popleft()
            near=sorted((abs(ts-t),ts,j) for j,(ts,rows) in enumerate(self.scans)
                        if abs(ts-t)<=c.pair_dt+1e-8 and len(rows))
            for _,ts,j in near:
                sref=tr.ref(ts,c.max_gap)
                if sref is not None:
                    break
            else:
                self.count["no_supported_scan"]+=1; continue
            img,mask,k=pixels(self.root,e)
            m.observe(Frame(t,iref,ts,sref,self.scans[j][1],img,mask,k))
            stats=m.cycle()
            self.log.write(json.dumps(dict(t=t,scan=ts,**stats))+"\n"); self.log.flush()
            self.count["mapped_inputs"]+=1

    def flush(self):
        self.drain(final=True)
        self.log.close()
        m=self.mapper
        write_ply(self.out/("gaussians_%03d.ply"%self.epoch),m)
        # Held-out images never entered candidate selection, birth or optimization.
        # Evaluate final-map RGB and coverage, never treat a blank black pixel as observed.
        se,n,valid_total=0.,0,0
        xyz,_,cov=m.world()
        for e,ref in self.held:
            img,mask,k=pixels(self.root,e)
            from .geometry import inv
            cam=m.ex@inv(m.trajectory.at(ref)); r=cam[:3,:3]
            pc=apply(cam,xyz); s=r@cov@r.T
            ids=m.visible(pc,s,k,img.shape[:2])
            valid_total+=int(mask.sum())
            if not len(ids):
                continue
            rgb,a,_=m.render(pc[ids],s[ids],np.array([m.map[i].color for i in ids]),k,*img.shape[:2])
            good=mask&(a>=m.c.coverage)
            se+=float(((rgb[good]-img[good])**2).sum())
            n+=int(good.sum())
        self.reports.append(dict(epoch=self.epoch,gaussians=len(m.map),events=dict(self.count),
                                 heldout_frames=len(self.held),covered_pixels=n,valid_pixels=valid_total,
                                 coverage=n/valid_total if valid_total else None,
                                 psnr_covered_db=-10*np.log10(max(1e-12,se/(3*n))) if n else None))
