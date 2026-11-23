import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
import numpy as np
from gs_mapping.geometry import Ref,Trajectory,apply,shape
from gs_mapping.mapper import Mapper,Gaussian,Frame
from gs_mapping.policy import Config,select
from gs_mapping.render import cpu,color_terms
from gs_mapping.demo import generate
from gs_mapping.replay import Session,records,write_ply


def knot(i,t,x=0.,p=1.,active=True):
    return dict(id=i,t=t,T=[x,0,0,0,0,0,1],P=(p*np.eye(6)).ravel().tolist(),active=active)


class MappingTests(unittest.TestCase):
    def test_brackets_covariance_and_retirement(self):
        tr=Trajectory(.2,4)
        tr.update([knot(1,0.),knot(2,1.,2)])
        mid=tr.ref(.5,1.1)
        self.assertIsNone(tr.ref(.5,.9)); self.assertIsNone(tr.ref(1.1,2.))
        np.testing.assert_allclose(tr.at(mid)[:3,3],[1,0,0])
        pt=np.array([2.,1.,3.]); one=Ref(1,1,0.)
        np.testing.assert_allclose(tr.relative_noise(mid,mid,pt),0.)
        self.assertGreater(tr.relative_noise(one,mid,pt)[1,1],0.)
        tr.update([knot(1,0.,.4,active=False),knot(2,1.,2.2)])
        np.testing.assert_allclose(tr.at(mid)[:3,3],[1.3,0,0])
        k=knot(3,2.,3.);k["P"]=None;tr.update([k])
        self.assertEqual(tr.missing_cov,1);self.assertGreater(tr.knots[3][2][0,0],0)

    def test_anchored_map_rotates_shape_and_mean(self):
        m=Mapper(Config());a=Ref(1,1,0.)
        m.trajectory.update([knot(1,0.)])
        m.map.append(Gaussian(a,np.zeros(3),np.array([1.,0,0]),np.array([1.,0,0]),np.ones(3)))
        p,_,c=m.world()
        k=knot(1,0.,1.,active=False);k["T"][5:]=[2**-.5,2**-.5]
        m.trajectory.update([k]);q,n,d=m.world()
        np.testing.assert_allclose(q,[[1,1,0]],atol=1e-9)
        np.testing.assert_allclose(n,[[0,1,0]],atol=1e-9)
        self.assertLess(d[0,1,1],d[0,0,0]);self.assertLess(c[0,0,0],c[0,1,1])

    def test_projection_downweights_uncertain_cross_pose(self):
        m=Mapper(Config());a,b=Ref(1,1,0.),Ref(2,2,0.)
        m.trajectory.update([knot(1,0.,p=.01),knot(2,.1,p=.01)])
        def frame(ref):
            return Frame(.1,ref,0.,a,np.array([[0.,0.,2.,0,0,-1,.001]]),
                         np.ones((12,12,3)),np.ones((12,12),bool),np.array([20.,20.,5.,5.]))
        same,other=frame(a),frame(b)
        m.prepare(same);m.prepare(other)
        self.assertEqual(same.reliability,1.0);self.assertLess(other.reliability,.8)
        m.c.use_pose_uncertainty=False;m.prepare(other)
        self.assertEqual(other.reliability,1.0)

    def test_overlap_cost_and_fixed_baseline(self):
        def frame(t,scan,cell):
            return SimpleNamespace(t=t,scan_id=scan,hits=[0]*2,
                 evidence={cell:(np.zeros((3,3)),np.diag([2.,0.,0.]))})
        a,b,d=frame(1,1,0),frame(2,1,0),frame(3,2,1)
        c=Config(gain_min=.3,max_work=4,max_frames=3)
        self.assertEqual({x.t for x in select([a,b,d],c)},{2,3})
        self.assertEqual(len(select([a,b],c)),1)
        c.fixed=True
        self.assertEqual([x.t for x in select([a,b,d],c)],[3,2])

    def test_multiscan_shape_and_ply_convention(self):
        m=Mapper(Config());a=Ref(1,1,0.);b=Ref(2,2,0.)
        m.trajectory.update([knot(1,0.),knot(2,.2)])
        g=Gaussian(a,np.zeros(3),np.zeros(3),np.array([0.,0.,1.]),np.ones(3)*.5)
        pts=[np.array([x,y,0.]) for x,y in [(0.,0.),(.1,0.),(0.,.08),(-.1,0.)]]
        obs=[(p,g.normal,a,.001) for p in pts]
        m.fit_shape(g,np.eye(4),obs);self.assertIsNone(g.local_cov)
        obs[0]=(pts[0],g.normal,b,.001)
        m.fit_shape(g,np.eye(4),obs);m.map.append(g)
        cov=m.world()[2][0];self.assertTrue((np.linalg.eigvalsh(cov)>0).all())
        self.assertFalse(np.allclose(cov,shape(g.normal,m.c.radius,m.c.thickness)))
        with TemporaryDirectory() as td:
            path=Path(td)/"m.ply";write_ply(path,m)
            vals=np.fromstring(path.read_text().splitlines()[-1],sep=" ")
            from scipy.spatial.transform import Rotation
            r=Rotation.from_quat(vals[[14,15,16,13]]).as_matrix()
            np.testing.assert_allclose((r*np.exp(2*vals[10:13]))@r.T,cov,atol=1e-8)

    def test_compositing_gradient_and_color_descent(self):
        m=np.array([[0.,0.,2.],[.01,.02,2.6]])
        s=np.stack([shape(np.array([0.,0,1]),.14,.025)]*2)
        k=[30.,33.,8.4,7.1];c=np.array([[.2,.3,.4],[.6,.4,.3]])
        col,alpha,back=cpu(m,s,c,k,16,19)
        rng=np.random.default_rng(3);g=rng.random(col.shape);delta=rng.normal(size=c.shape)
        fd=(cpu(m,s,c+1e-5*delta,k,16,19)[0]-cpu(m,s,c-1e-5*delta,k,16,19)[0])/2e-5
        self.assertAlmostEqual(np.sum(fd*g),np.sum(delta*back(g)),places=8)
        gt=cpu(m,s,c*.6,k,16,19)[0];mask=np.ones(alpha.shape,bool)
        grad,den,e,_=color_terms(cpu,m,s,c,k,gt,mask,.08)
        _,_,next_e,_=color_terms(cpu,m,s,c-grad/np.maximum(den,1e-9),k,gt,mask,.08)
        self.assertLess(next_e,e)

    def test_point_plane_descent_and_repeated_scan(self):
        m=Mapper(Config(max_frames=3,iterations=2,budget_ms=10000,gain_min=.001))
        m.trajectory.update([knot(1,0.,p=.00001),knot(2,.1,p=.00001)])
        ray=np.array([0.,0.,-1]);local=np.array([0.,0.,2.])
        m.map.append(Gaussian(Ref(1,1,0.),local.copy(),local.copy(),ray,np.ones(3)*.3))
        for t in (.09,.1):
            m.observe(Frame(t,Ref(2,2,0.),.1,Ref(2,2,0.),np.array([[0,0,2.06,0,0,-1,.0002]]),
                            np.ones((17,17,3))*.3,np.ones((17,17),bool),np.array([20,20,8,8])))
        r=m.cycle()
        self.assertEqual(len(m.map),1);self.assertEqual(r["geometry_terms"],1)
        self.assertLess(abs(m.map[0].mean[2]-2.06),.03)

    def test_journal_end_to_end_and_reset(self):
        with TemporaryDirectory() as parent:
            root=Path(parent)/"in";out=Path(parent)/"out";out.mkdir()
            generate(root);c=Config(max_frames=2,max_active=5,max_map=30,max_work=35,iterations=1)
            s=Session(root,out,c,"cpu",3)
            for e in records(root/"events.jsonl"): s.event(e)
            s.event({"kind":"reset","epoch":1})
            s.flush()
            first,last=s.reports
            self.assertGreater(first["gaussians"],0);self.assertLessEqual(first["gaussians"],30)
            self.assertEqual(last["gaussians"],0);self.assertEqual(first["heldout_frames"],2)
            self.assertGreater(first["covered_pixels"],0)
            import json
            stats=[json.loads(l) for l in (out/"window_000.jsonl").read_text().splitlines()]
            self.assertTrue(all(x["active"]<=5 and x["work"]<=35 and len(x["selected"])<=2 for x in stats))
            self.assertTrue(all(abs(x["t"]-x["scan"])>1e-4 for x in stats))
            self.assertFalse(any(t in {0.275,0.635} for x in stats for t in x["selected"]))
            self.assertIn("element vertex "+str(first["gaussians"]),(out/"gaussians_000.ply").read_text())

    @unittest.skipUnless(os.environ.get("GS_TEST_CUDA")=="1","requires pinned GS-LIVO + NVIDIA CUDA")
    def test_upstream_cuda_matches_cpu(self):
        from gs_mapping.gs_livo_cuda import render
        m=np.array([[0.,0.,2.],[.02,.01,2.5]]);c=np.array([[.6,.1,.2],[.2,.7,.3]])
        s=np.stack([shape(np.array([0.,0.,1.]),.14,.03)]*2);k=[26.,23.,6.1,9.2]
        a,aa,av=cpu(m,s,c,k,17,18);b,ba,bv=render(m,s,c,k,17,18)
        np.testing.assert_allclose(a,b,atol=.012);np.testing.assert_allclose(aa,ba,atol=.015)
        g=np.ones_like(a)
        np.testing.assert_allclose(av(g),bv(g),rtol=.025,atol=.03)


if __name__=="__main__":
    unittest.main()
