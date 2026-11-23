"""坐标与不确定性。T_AB 将 B 中的点映射到 A，状态切空间为 [dp_W, dtheta_I]。"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def transform(values):
    a = np.asarray(values, float).reshape(7)
    if not np.isfinite(a).all():
        raise ValueError("non-finite transform")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(a[3:]).as_matrix()
    t[:3, 3] = a[:3]
    return t


def apply(t, xyz):
    return np.asarray(xyz) @ t[:3, :3].T + t[:3, 3]


def inv(t):
    r = t[:3, :3].T
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, -(r @ t[:3, 3])
    return out


def skew(x):
    a, b, c = x
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])


@dataclass(frozen=True)
class Ref:
    a: int
    b: int
    u: float

    def weights(self):
        return {self.a: 1.0} if self.a == self.b else {self.a: 1-self.u, self.b: self.u}


class Trajectory:
    def __init__(self, fallback_p, fallback_r):
        self.knots = {}
        self.fallback = np.diag([fallback_p**2]*3 + [np.deg2rad(fallback_r)**2]*3)
        self.missing_cov = 0

    def update(self, poses):
        self._cache = {}
        for frame in poses:
            i = int(frame["id"])
            t = float(frame["t"])
            if not np.isfinite(t) or (i in self.knots and abs(self.knots[i][0]-t) > 1e-7):
                raise ValueError("pose id changed timestamp")
            p = frame["P"]
            if p is not None:
                p = np.asarray(p, float).reshape(6, 6)
                p = .5*(p+p.T)
                if not np.isfinite(p).all() or np.linalg.eigvalsh(p)[0] < -1e-9:
                    p = None
            if p is None:
                self.missing_cov += 1
                p = self.fallback
            self.knots[i] = (t, transform(frame["T"]), p, bool(frame["active"]))
        self.ordered = sorted((v[0], i) for i, v in self.knots.items())
        self.times = np.array([x[0] for x in self.ordered])

    def ref(self, t, max_gap):
        if not self.knots:
            return None
        j = int(np.searchsorted(self.times, t))
        for k in (j, j-1):
            if 0 <= k < len(self.times) and abs(self.times[k]-t) <= 1e-7:
                i = self.ordered[k][1]
                return Ref(i, i, 0.)
        if not 0 < j < len(self.times):
            return None
        ta, a = self.ordered[j-1]
        tb, b = self.ordered[j]
        return Ref(a, b, (t-ta)/(tb-ta)) if tb-ta <= max_gap else None

    def at(self, ref):
        if ref not in self._cache:
            a, b = self.knots[ref.a][1], self.knots[ref.b][1]
            if ref.a == ref.b:
                out = a
            else:
                out = np.eye(4)
                r = Rotation.from_matrix(a[:3,:3].T @ b[:3,:3]).as_rotvec()
                out[:3,:3] = a[:3,:3] @ Rotation.from_rotvec(ref.u*r).as_matrix()
                out[:3,3] = (1-ref.u)*a[:3,3]+ref.u*b[:3,3]
            self._cache[ref] = out
        return self._cache[ref]

    def relative_noise(self, a, b, world_point):
        """世界坐标下的相对点误差。相同锚点的扰动严格抵消；不同结点忽略交叉协方差。"""
        if a == b:
            return np.zeros((3, 3))
        wa, wb = a.weights(), b.weights()
        pa, pb = self.at(a)[:3,3], self.at(b)[:3,3]
        cov = np.zeros((3,3))
        for i in wa.keys() | wb.keys():
            la, lb = wa.get(i, 0.), wb.get(i, 0.)
            _, k, p, _ = self.knots[i]
            j = np.column_stack(((la-lb)*np.eye(3),
                                 (-la*skew(world_point-pa)+lb*skew(world_point-pb)) @ k[:3,:3]))
            cov += j @ p @ j.T
        return .5*(cov+cov.T)


def pixel(k, pc):
    fx, fy, cx, cy = k
    x, y, z = pc
    j = np.array([[fx/z, 0, -fx*x/z**2], [0, fy/z, -fy*y/z**2]])
    return np.array([fx*x/z+cx, fy*y/z+cy]), j


def shape(normal, radius, thickness):
    n = np.asarray(normal)
    return radius**2*np.eye(3)+(thickness**2-radius**2)*np.outer(n, n)
