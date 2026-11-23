"""独立建图窗。只读取优化位姿；高斯的均值和形状一直保存在各自的锚点坐标系。"""
from dataclasses import dataclass, field
from itertools import product
from time import perf_counter
import numpy as np
from .geometry import Trajectory, apply, inv, pixel, shape
from .policy import select
from .render import renderer, color_terms

NEIGHBOURS = tuple(product((-1,0,1),repeat=3))


@dataclass
class Gaussian:
    anchor: object
    seed: np.ndarray
    mean: np.ndarray
    normal: np.ndarray
    color: np.ndarray
    local_cov: object = None
    views: dict = field(default_factory=dict)  # bounded angular bins, expressed in anchor I


@dataclass
class Frame:
    t: float
    ref: object
    scan_id: float
    scan_ref: object
    rows: np.ndarray
    image: np.ndarray
    mask: np.ndarray
    k: np.ndarray
    hits: list = None
    evidence: dict = None
    reliability: float = 0.


class Mapper:
    def __init__(self, cfg, backend="cpu"):
        self.c = cfg.check()
        self.trajectory = Trajectory(cfg.pose_sigma_m,cfg.pose_sigma_deg)
        self.render = renderer(backend)
        self.ex = np.eye(4)  # T_CI
        self.frames, self.map, self.birth = [], [], {}
        self.last = {}
        self.selections = 0

    def observe(self, frame):
        self.frames.append(frame)
        self.frames = [f for f in self.frames if f.t >= frame.t-self.c.history][-self.c.max_candidates:]

    def camera(self, f):
        return self.ex @ inv(self.trajectory.at(f.ref))

    def prepare(self, f):
        """扫描和曝光使用不同的优化位姿；按像素深度筛掉遮挡点，构建候选帧证据。"""
        tr,c = self.trajectory,self.c
        camera, scan = self.camera(f), tr.at(f.scan_ref)
        center = inv(camera)[:3,3]
        closest = {}
        for idx,row in enumerate(f.rows):
            pw = apply(scan,row[:3])
            pc = apply(camera,pw)
            if pc[2] <= .2:
                continue
            uv,j = pixel(f.k,pc)
            x,y = np.rint(uv).astype(int)
            h,w = f.image.shape[:2]
            if not (1 <= x < w-1 and 1 <= y < h-1 and f.mask[y,x]):
                continue
            if (x,y) in closest and closest[x,y][0] <= pc[2]:
                continue
            closest[x,y] = (pc[2],idx,pw,j)
        pts = sorted(closest.items(),key=lambda kv: kv[1][1])
        # Evenly subsample the surviving returns. Never let an oversized first
        # candidate consume work that can never fit the configured budget.
        limit = min(c.max_pairs,c.max_work)
        if len(pts)>limit:
            pts = [pts[k] for k in np.linspace(0,len(pts)-1,limit,dtype=int)]
        f.hits, f.evidence, reliabilities = [], {}, []
        for (x,y),(_,i,pw,j) in pts:
            row = f.rows[i]
            n = scan[:3,:3]@row[3:6]
            noise = tr.relative_noise(f.scan_ref,f.ref,pw) if c.use_pose_uncertainty else np.zeros((3,3))
            pj = j@camera[:3,:3]
            q = 1/(1+max(0.,np.trace(pj@noise@pj.T))/(2*c.pixel_sigma**2))
            li = c.plane_sigma**2/(c.plane_sigma**2+row[6])
            ray = pw-center
            ray /= np.linalg.norm(ray)
            tex = min(1.,4*(np.linalg.norm(f.image[y,x+1]-f.image[y,x-1])+
                            np.linalg.norm(f.image[y+1,x]-f.image[y-1,x])))
            block = tuple(np.floor(pw/c.block).astype(int))
            # Per-cell average prevents many pixels in one cell dominating lag selection.
            if block not in f.evidence:
                f.evidence[block] = [np.zeros((3,3)),np.zeros((3,3)),0]
            v = f.evidence[block]
            v[0] += q*(.05+tex)*(np.eye(3)-np.outer(ray,ray))
            v[1] += li*np.outer(n,n)
            v[2] += 1
            f.hits.append((i,pw,n,float(q),x,y))
            reliabilities.append(q)
        f.evidence = {k:(a/count,b/count) for k,(a,b,count) in f.evidence.items()}
        f.reliability = float(np.mean(reliabilities)) if reliabilities else 0.

    def world(self):
        if not self.map:
            return np.empty((0,3)),np.empty((0,3)),np.empty((0,3,3))
        p,n,s = [],[],[]
        for g in self.map:
            t = self.trajectory.at(g.anchor)
            p.append(apply(t,g.mean))
            n.append(t[:3,:3]@g.normal)
            local = g.local_cov if g.local_cov is not None else shape(g.normal,self.c.radius,self.c.thickness)
            s.append(t[:3,:3]@local@t[:3,:3].T)
        return np.asarray(p),np.asarray(n),np.asarray(s)

    def cycle(self):
        start = perf_counter()
        c,tr = self.c,self.trajectory
        for f in self.frames:
            self.prepare(f)
        frames = select(self.frames,c)
        world, normals, _ = self.world()
        wp,wn = list(world),list(normals)
        key = lambda p: tuple(np.floor(p/c.voxel).astype(int))
        bins = {}
        for idx,p in enumerate(wp):
            bins.setdefault(key(p),[]).append(idx)
        geom,seen,activity,bins_now,raw_error = {},set(),{},{},{}
        eye=np.eye(3)
        centers={f.t:inv(self.camera(f))[:3,3] for f in frames}
        births=0
        observations = [(q,f,i,p,n,x,y) for f in frames for i,p,n,q,x,y in f.hits]
        for q,f,i,p,n,x,y in sorted(observations,key=lambda e:-e[0]):
            cell = key(p)
            obs_id = (f.scan_id,i)
            idx = self.birth.get(obs_id)
            if idx is None:
                choices = []
                for d in NEIGHBOURS:
                    for j in bins.get(tuple(a+b for a,b in zip(cell,d)),()):
                        if abs(n@wn[j])>.8:
                            dist = np.linalg.norm(wp[j]-p)
                            if dist<c.voxel:
                                choices.append((dist,j))
                if choices:
                    idx = min(choices)[1]
                elif len(self.map)<c.max_map and births<c.max_active:
                    row = f.rows[i]
                    self.map.append(Gaussian(f.scan_ref,row[:3].copy(),row[:3].copy(),
                                             row[3:6].copy(),f.image[y,x].copy()))
                    idx = self.birth[obs_id] = len(self.map)-1
                    bins.setdefault(cell,[]).append(idx); wp.append(p); wn.append(n); births+=1
            if idx is None:
                continue
            g=self.map[idx]; r=tr.at(g.anchor)[:3,:3]
            ray=r.T@(p-centers[f.t]); ray/=np.linalg.norm(ray)
            normal=r.T@n
            if normal[np.argmax(np.abs(normal))]<0:
                normal=-normal
            evidence=bins_now.setdefault(idx,{})
            for modality,vec,hess in (("v",ray,q*(eye-np.outer(ray,ray))),
                                      ("l",normal,c.plane_sigma**2/(c.plane_sigma**2+f.rows[i,6])*np.outer(normal,normal))):
                bucket=(modality,)+tuple(np.rint(2*vec).astype(int))
                # At most one descriptor per angular bin. Never credit repeat views as independent.
                old=evidence.get(bucket)
                if old is None or np.trace(hess)>np.trace(old):
                    evidence[bucket]=hess
            raw_error[idx]=max(raw_error.get(idx,0.),q*np.mean(abs(g.color-f.image[y,x])))
            if (obs_id,idx) not in seen:
                seen.add((obs_id,idx))
                geom.setdefault(idx,[]).append((p,n,f.scan_ref,f.rows[i,6]))
        for idx, bins_seen in bins_now.items():
            seen_before=self.map[idx].views
            h0=eye+sum(seen_before.values())
            new=sum(v for key_,v in bins_seen.items() if key_ not in seen_before)
            gain=np.linalg.slogdet(h0+new)[1]-np.linalg.slogdet(h0)[1]
            geom_error=max((abs(n@(wp[idx]-p))/c.voxel for p,n,_,_ in geom.get(idx,())),default=0.)
            score=gain+raw_error[idx]+min(1.,geom_error)
            if c.fixed or c.fixed_splats or score>=c.splat_gain:
                activity[idx]=score
        active=sorted(activity,key=(lambda i:i) if (c.fixed or c.fixed_splats) else (lambda i:(-activity[i],i)))[:c.max_active]
        colors = np.array([g.color for g in self.map])
        initial, final, steps = 0.,0.,0
        for _ in range(c.iterations):
            if not active or (steps and (perf_counter()-start)*1000>=c.budget_ms):
                break
            # IRLS point-to-plane step in the Gaussian's original local tangent.
            for idx in active:
                g=self.map[idx]
                anchor=tr.at(g.anchor)
                point=apply(anchor,g.mean)
                h=np.eye(3)/(c.radius**2)
                b=(g.mean-g.seed)/(c.radius**2)
                for p,n,sref,var in geom.get(idx,()):
                    pose_var=n@tr.relative_noise(g.anchor,sref,point)@n if c.use_pose_uncertainty else 0.
                    sig2 = c.plane_sigma**2+var+max(0.,pose_var)
                    residual=n@(point-p)
                    w=min(1.,2*np.sqrt(sig2)/max(1e-9,abs(residual)))/sig2
                    a=anchor[:3,:3].T@n
                    h+=w*np.outer(a,a); b+=w*a*residual
                g.mean-=np.linalg.solve(h,b)
                if c.learn_shape:
                    self.fit_shape(g,anchor,geom.get(idx,()))
            xyz,_,s = self.world()
            grad,den=np.zeros_like(colors),np.zeros_like(colors)
            err,count=0.,0
            for f in frames:
                cam=self.camera(f); r=cam[:3,:3]
                pc=apply(cam,xyz); cov=r@s@r.T
                # Screen cull before invoking either renderer; inactive visible
                # Gaussians still occlude and contribute to the residual.
                ids=self.visible(pc,cov,f.k,f.image.shape[:2])
                if not len(ids):
                    continue
                a,b,d,k=color_terms(self.render,pc[ids],cov[ids],colors[ids],f.k,
                                    f.image,f.mask,c.coverage)
                grad[ids]+=f.reliability*a; den[ids]+=f.reliability*b
                err+=d; count+=k
            if not steps:
                initial=err/max(1,count)
            final=err/max(1,count)
            colors[active]=np.maximum(0.,np.minimum(1.,colors[active]-
                                      grad[active]/np.maximum(den[active],1e-9)))
            steps+=1
        for idx in active:
            self.map[idx].color=colors[idx]
            for k,v in bins_now[idx].items():
                self.map[idx].views.setdefault(k,v)
        self.selections+=1
        self.last=dict(selected=[f.t for f in frames],candidate=len(self.frames),active=len(active),
                       gaussians=len(self.map),births=births,geometry_terms=sum(len(geom.get(i,())) for i in active),
                       steps=steps,mae_before_first=initial,mae_before_last=final,
                       work=sum(len(f.hits) for f in frames),elapsed_ms=(perf_counter()-start)*1000,
                       missing_cov_blocks=tr.missing_cov)
        return self.last

    def fit_shape(self,g,anchor,obs):
        """在锚点内对已去重的多扫描回波作带先验矩估计；共线时保持现有椭球。"""
        if len(obs)<3 or len({s for _,_,s,_ in obs})<2:
            return
        local=apply(inv(anchor),np.array([p for p,_,_,_ in obs]))
        noise=[var+(max(0.,n@self.trajectory.relative_noise(g.anchor,ref,p)@n)
                    if self.c.use_pose_uncertainty else 0.) for p,n,ref,var in obs]
        weights=1/(self.c.plane_sigma**2+np.array(noise)); weights/=weights.sum()
        d=local-np.average(local,axis=0,weights=weights)
        scatter=(d.T*weights)@d
        if np.linalg.eigvalsh(scatter)[1]<self.c.thickness**2*.1:
            return
        prior=shape(g.normal,self.c.radius,self.c.thickness)
        cov=(2*prior+len(obs)*scatter)/(2+len(obs))
        vals,vec=np.linalg.eigh(cov)
        vals=np.minimum((2*self.c.radius)**2,np.maximum((.5*self.c.thickness)**2,vals))
        g.local_cov=(vec*vals)@vec.T

    @staticmethod
    def visible(pc,cov,k,hw):
        ids=np.flatnonzero(pc[:,2]>.2)
        if not len(ids):
            return ids
        fx,fy,cx,cy=k
        v=pc[ids]; z=v[:,2]
        x,y=fx*v[:,0]/z+cx,fy*v[:,1]/z+cy
        # Conservative footprint bound from max projected Jacobian norm.
        bound=3*np.sqrt(.3+np.linalg.eigvalsh(cov[ids])[:,-1]*(fx*fx*(1+(v[:,0]/z)**2)
                    +fy*fy*(1+(v[:,1]/z)**2))/z**2)
        return ids[(x+bound>=0)&(x-bound<hw[1])&(y+bound>=0)&(y-bound<hw[0])]
