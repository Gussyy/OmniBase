"""A serial kinematic chain: forward kinematics, Jacobian, and a batched inverse solve.

Deliberately not tied to any one robot. Give it a list of joints -- each an origin, a rotation,
an axis and a pair of stops -- and it will tell you, for a whole array of 6-DoF target poses at
once, the best joint angles and how far short they fall. That last part is the point: this
library exists to measure what an arm *cannot* do with a recording, so a solver that quietly
reported success would be worse than useless.

The inverse solve is damped least squares rather than a closed form. A closed form is faster,
but it has to be derived per robot and it only exists for some of them -- the arm this was
written for happens to admit one, and the second arm in the same repository does not. One
general solver that works on anything you can describe beats two special ones.

Conventions, because they are the thing that silently breaks:

* positions are metres, angles radians;
* quaternions are ``(x, y, z, w)`` -- scipy's order -- everywhere in the public API;
* a joint's ``rot`` is its frame relative to its parent, and its ``axis`` is expressed in that
  joint frame, which is what USD's ``localRot0``/``physics:axis`` and URDF's ``origin``/``axis``
  both give you;
* the tool point is an offset in the last link's frame, so it is the point between the fingers
  rather than the wrist flange.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

_AXES = {"x": 0, "y": 1, "z": 2, "X": 0, "Y": 1, "Z": 2, 0: 0, 1: 1, 2: 2}


@dataclass(frozen=True)
class Joint:
    """One revolute joint.

    Args:
        origin: (3,) translation of the joint frame in the parent link's frame.
        rot: rotation of the joint frame in the parent link's frame, as ``(x, y, z, w)``.
        axis: which of the joint frame's own axes it turns about -- ``"x"``, ``"y"`` or ``"z"``.
        lower, upper: the stops, in radians.
        name: for error messages and for mapping onto a robot's joint ordering.
    """

    origin: tuple
    rot: tuple
    axis: str
    lower: float
    upper: float
    name: str = ""

    @staticmethod
    def from_usd(origin, local_rot_wxyz, axis, lower_deg, upper_deg, name=""):
        """A joint as a USD ``PhysicsRevoluteJoint`` writes it: quaternion (w, x, y, z), degrees."""
        w, x, y, z = local_rot_wxyz
        return Joint(tuple(origin), (x, y, z, w), axis,
                     np.radians(lower_deg), np.radians(upper_deg), name)

    @staticmethod
    def from_urdf(xyz, rpy, axis_xyz, lower, upper, name=""):
        """A joint as a URDF writes it: fixed-axis roll-pitch-yaw, and an arbitrary axis vector.

        URDF lets the axis be any unit vector; this chain wants a cardinal one, so the axis is
        folded into the joint's own rotation and the joint is then declared to turn about z.
        """
        a = np.asarray(axis_xyz, dtype=float)
        a = a / np.linalg.norm(a)
        base = R.from_euler("xyz", rpy)
        # Rotation taking +z onto the requested axis, so the joint can spin about z.
        v = np.cross([0.0, 0.0, 1.0], a)
        c = float(np.dot([0.0, 0.0, 1.0], a))
        if np.linalg.norm(v) < 1e-12:
            extra = R.identity() if c > 0 else R.from_rotvec([np.pi, 0.0, 0.0])
        else:
            extra = R.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1.0, 1.0)))
        return Joint(tuple(xyz), tuple((base * extra).as_quat()), "z", lower, upper, name)


def _turn(M, axis, c, s):
    """``M @ R(axis, q)`` for a cardinal ``axis``, given ``cos q``/``sin q`` as (..., 1) columns.

    Overwrites ``M``.

    A rotation about one of the frame's own axes leaves that column of M alone and mixes the
    other two, so the (N, 3, 3) @ (N, 3, 3) product is four multiplies wide, not twenty-seven --
    and the rotation matrix itself, which a placement sweep would otherwise assemble a few
    hundred million times, never has to exist.
    """
    b, d = (axis + 1) % 3, (axis + 2) % 3
    Mb, Md = M[..., b].copy(), M[..., d]
    M[..., b] = c * Mb + s * Md          # in place: every caller passes a fresh product, and
    M[..., d] = c * Md - s * Mb          # column ``axis`` comes through untouched anyway
    return M


def _rot_err(M, Rt):
    """Angle between two batches of rotation matrices, radians."""
    return np.arccos(np.clip((np.einsum("...ij,...ij->...", M, Rt) - 1.0) / 2.0, -1.0, 1.0))


def _chol3(A, v):
    """``A^-1 v`` for symmetric positive-definite (..., 3, 3) ``A`` and (..., 3, 1) ``v``.

    Every ``A`` this is handed is ``J J^T + d^2 I`` with ``d > 0``, which is symmetric positive
    definite by construction, so a Cholesky factorisation applies and there is no pivoting to do.
    Written out for 3x3 because the batch is hundreds of thousands of them and
    ``np.linalg.solve`` is a LAPACK call per matrix: measured on a real sweep, this is 20% faster
    than ``np.linalg.solve`` and reaches the same accuracy.

    Cramer's rule is a third option and is a trap. It is the same speed as this (measured: 1.5%
    slower), but its determinant is a sum of six products that cancel as ``A`` approaches
    singular. On ill-conditioned batches at the damping this library uses it gives a backward
    error of 1.3e-07 against this routine's 2.1e-16 -- nine orders -- and near-singular enough it
    returns ``nan`` rather than raising, which reads downstream as "the arm cannot reach", the
    one answer this library must never invent. Cholesky's pivots are square roots of quantities
    the damping keeps positive, so it has no such failure.
    """
    a = A[..., 0, 0]; b = A[..., 0, 1]; c = A[..., 0, 2]
    e = A[..., 1, 1]; f = A[..., 1, 2]; i = A[..., 2, 2]
    l00 = np.sqrt(a)
    l10 = b / l00
    l20 = c / l00
    l11 = np.sqrt(e - l10 * l10)
    l21 = (f - l20 * l10) / l11
    l22 = np.sqrt(i - l20 * l20 - l21 * l21)
    y0 = v[..., 0, 0] / l00                                  # forward substitution, L y = v
    y1 = (v[..., 1, 0] - l10 * y0) / l11
    y2 = (v[..., 2, 0] - l20 * y0 - l21 * y1) / l22
    out = np.empty_like(v)
    x2 = y2 / l22                                            # back substitution, L^T x = y
    x1 = (y1 - l21 * x2) / l11
    out[..., 2, 0] = x2
    out[..., 1, 0] = x1
    out[..., 0, 0] = (y0 - l10 * x1 - l20 * x2) / l00
    return out


def _dilate(m, k):
    """Grow a boolean raster by ``k`` cells in every direction. Separable, so 2k passes."""
    for ax in (0, 1):
        m = np.swapaxes(m, 0, ax)
        out = m.copy()
        for s in range(1, k + 1):
            out[s:] |= m[:-s]
            out[:-s] |= m[s:]
        m = np.swapaxes(out, 0, ax)
    return m


def _rotvec(A):
    """Axis-angle vector of rotation matrices (..., 3, 3). scipy would dominate the loop."""
    w = 0.5 * np.stack([A[..., 2, 1] - A[..., 1, 2],
                        A[..., 0, 2] - A[..., 2, 0],
                        A[..., 1, 0] - A[..., 0, 1]], axis=-1)
    s = np.linalg.norm(w, axis=-1, keepdims=True)
    c = ((A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]) - 1.0)[..., None] / 2.0
    theta = np.arctan2(s, np.clip(c, -1.0, 1.0))
    return np.where(s > 1e-9, w / np.where(s > 1e-9, s, 1.0) * theta, w)


class Chain:
    """A serial arm, with the tool point that actually grasps.

    Args:
        joints: the chain, base outwards.
        tool: (3,) offset from the last link's frame to the point between the fingers.
        name: for messages.
    """

    def __init__(self, joints, tool=(0.0, 0.0, 0.0), name="chain"):
        self.joints = list(joints)
        self.name = name
        self.tool = np.asarray(tool, dtype=float)
        self.origin = np.array([j.origin for j in self.joints], dtype=float)
        self.rot = np.stack([R.from_quat(j.rot).as_matrix() for j in self.joints])
        self.axis = [_AXES[j.axis] for j in self.joints]
        self.limits = np.array([[j.lower, j.upper] for j in self.joints], dtype=float)
        if np.linalg.norm(self.tool) > 0:
            self.tool_dir = self.tool / np.linalg.norm(self.tool)
        else:
            self.tool_dir = np.array([0.0, 0.0, 1.0])

    @property
    def n(self):
        return len(self.joints)

    # ---- forward ---------------------------------------------------------------------------
    def fk(self, q):
        """Tool point and orientation in the base frame. ``q`` is (..., n) -> ((..., 3), (..., 3, 3))."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        p, M = self._walk(q, self.n)
        return p + np.einsum("...ik,k->...i", M, self.tool), M

    def _walk(self, q, stop):
        """Frame after the first ``stop`` joints -- ``(p, M)``, the guts of :meth:`fk`."""
        p = np.zeros(q.shape[:-1] + (3,))
        M = np.broadcast_to(np.eye(3), q.shape[:-1] + (3, 3)).copy()
        C, S = np.cos(q[..., None]), np.sin(q[..., None])   # all joints at once, once
        for i in range(stop):
            p = p + np.einsum("...ik,k->...i", M, self.origin[i])
            M = _turn(np.einsum("...ik,kj->...ij", M, self.rot[i]),
                      self.axis[i], C[..., i, :], S[..., i, :])
        return p, M

    def links(self, q):
        """Joint origins plus the tool point, (..., n+1, 3). For drawing the arm, nothing else."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        p = np.zeros(q.shape[:-1] + (3,))
        M = np.broadcast_to(np.eye(3), q.shape[:-1] + (3, 3)).copy()
        C, S = np.cos(q[..., None]), np.sin(q[..., None])
        out = []
        for i in range(self.n):
            p = p + np.einsum("...ik,k->...i", M, self.origin[i])
            M = _turn(np.einsum("...ik,kj->...ij", M, self.rot[i]),
                      self.axis[i], C[..., i, :], S[..., i, :])
            out.append(p.copy())
        out.append(p + np.einsum("...ik,k->...i", M, self.tool))
        return np.stack(out, axis=-2)

    def jacobian(self, q):
        """``(tip, orientation, Jv, Jw)``. Column i is ``(axis_i x (tip - origin_i), axis_i)``."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        m, n = q.shape[0], self.n
        p = np.zeros((m, 3))
        M = np.broadcast_to(np.eye(3), (m, 3, 3)).copy()
        # Jw is built in place; ``M @ e_k`` is just column k of M, and a matmul against a basis
        # vector is a lot of arithmetic to arrive at a slice.
        Jw = np.empty((m, 3, n))
        orig = np.empty((m, 3, n))
        C, S = np.cos(q[:, None, :]), np.sin(q[:, None, :])
        for i in range(n):
            p = p + np.einsum("mik,k->mi", M, self.origin[i])
            M = np.einsum("mik,kj->mij", M, self.rot[i])
            Jw[:, :, i] = M[:, :, self.axis[i]]
            orig[:, :, i] = p
            M = _turn(M, self.axis[i], C[:, :, i], S[:, :, i])
        tip = p + np.einsum("mik,k->mi", M, self.tool)
        d = tip[:, :, None] - orig
        Jv = np.empty((m, 3, n))                 # np.cross written out: same three products,
        Jv[:, 0] = Jw[:, 1] * d[:, 2] - Jw[:, 2] * d[:, 1]   # without the axis bookkeeping it
        Jv[:, 1] = Jw[:, 2] * d[:, 0] - Jw[:, 0] * d[:, 2]   # does per call.
        Jv[:, 2] = Jw[:, 0] * d[:, 1] - Jw[:, 1] * d[:, 0]
        return tip, M, Jv, Jw

    # ---- cheap necessary conditions --------------------------------------------------------
    #: Most sample rows :meth:`envelope` will build. The sample grid is ``grid ** (joints that
    #: move the tool)``, so the default 96 is 884,736 rows on a 3-variable arm like the SO-101 and
    #: 85 million on a 4-variable one -- 15 GB, and a 5-variable arm asks for 60 GB and dies. The
    #: grid is therefore coarsened to fit this budget. Doing so is safe rather than merely
    #: convenient: a coarser grid only WIDENS the slack bound, so the region stays a superset of
    #: what the arm can reach and being outside it is still proof. It just prunes less.
    ROWS = 2_000_000

    def envelope(self, grid=96, cell=0.005, probes=(0.0, 0.125, 0.25)):
        """Where the tool CAN be, coarsely, as arithmetic you can run on millions of targets.

        Most of a placement sweep is hopeless: the grid is a metre across and the arm is not.
        Discovering that with a full iterative solve is the most expensive way to learn it, so
        this precomputes necessary conditions and :meth:`cannot_reach` applies them in bulk.

        Joint 0 does not move the tool through the workspace; it spins the whole arm about a line
        that is fixed in the base frame. So the set of reachable tool POINTS is a solid of
        revolution about that line, and (distance from the line, height along it) says everything
        a position can say. That two-dimensional region is sampled on a joint grid and rasterised.

        The same is done for a few points rigidly attached further along the tool's own axis. A
        point 100 mm past the fingertips is reachable only if the gripper both gets there and
        points a way the arm can support, so its envelope tests the orientation too -- at the
        cost of a wider tolerance, since an angular error is a bigger position error the further
        out you look. Then one scalar on top: ``Q = (M . tool_dir) . (axis x (tip - point))`` is
        also invariant under joint 0, and on an arm whose remaining joints swing in a plane it is
        nearly zero, because the tool points along the arm's own plane. A target held sideways to
        that plane cannot be met however far the base moves. Chains with no such structure simply
        get a large bound here and the condition never fires.

        Every threshold is widened by ``slack``: a Taylor bound, from gradients measured on the
        grid and a crude bound on the curvature, on how far the truth can sit from the nearest
        sample. So each region is a SUPERSET of the real one, and being outside it is proof.

        Args:
            grid: samples per joint, as a ceiling -- it is lowered to keep the sample count
                under :data:`ROWS`, which is what stops a 6-DoF arm asking for 60 GB. Finer is
                tighter and slower, and the cost is paid once and cached; past about 100 the
                tolerances dominate and it stops buying anything.
            cell: raster resolution of the (radius, height) regions, metres.
            probes: extra test points, as fractions of the arm's reach along the tool axis.

        Returns:
            dict, cached on the chain: ``axis`` and ``point`` (the line joint 0 spins about),
            ``maps`` (one raster per probe), ``qmax``, ``reach``.
        """
        key = (grid, cell, tuple(probes))
        env = getattr(self, "_envelope_cache", None)
        if env is not None and env["key"] == key:
            return env
        a = self.rot[0] @ np.eye(3)[self.axis[0]]
        a = a / np.linalg.norm(a)
        c = self.origin[0]
        # Curvature bound: every extra derivative of the chain is another rotation of something
        # no longer than what is left of it past the joint being turned.
        tail = np.array([np.linalg.norm(self.origin[i + 1:], axis=1).sum()
                         + np.linalg.norm(self.tool) for i in range(self.n)])
        curve = float(tail.max())
        free = set(self.free_joints())
        var = [i for i in range(1, self.n) if i not in free]   # joint 0: see the docstring
        if var:                                   # coarsen to fit ROWS; see the constant
            grid = max(3, min(grid, int(self.ROWS ** (1.0 / len(var)))))
        if grid ** max(len(var), 1) > 8 * self.ROWS:
            # Enough joints that even the coarsest useful grid is too big. A prune that cannot be
            # afforded is simply not applied -- the solver still gives the right answer, slowly.
            env = dict(key=key, cell=cell, axis=a, point=c.copy(), maps=[], qmax=None, reach=0.0)
            self._envelope_cache = env
            return env
        step = np.array([(self.limits[i, 1] - self.limits[i, 0]) / (grid - 1.0) for i in var])
        half = step / 2.0                         # worst case distance from a sample, per joint
        drift = float(half.sum()) ** 2

        mesh = [m.ravel() for m in
                np.meshgrid(*[np.linspace(self.limits[i, 0], self.limits[i, 1], grid)
                              for i in var], indexing="ij")]
        q = np.broadcast_to(self.limits.mean(axis=1),
                            (mesh[0].size if mesh else 1, self.n)).copy()
        for k, i in enumerate(var):
            q[:, i] = mesh[k]
        del mesh

        P, W, qv = [], [], []
        gx = np.zeros(len(var))                   # max ||d tip / d q_i|| over the grid
        gq = np.zeros(len(var))                   # max |dQ / d q_i| over the grid
        for lo in range(0, len(q), 250_000):
            p, M, Jv, Jw = self.jacobian(q[lo:lo + 250_000])
            w = M @ self.tool_dir
            u = np.cross(a, p - c)
            P.append(p)
            W.append(w)
            qv.append((w * u).sum(-1))
            Jv, Jw = Jv[:, :, var], Jw[:, :, var]
            gx = np.maximum(gx, np.linalg.norm(Jv, axis=1).max(axis=0))
            gq = np.maximum(gq, np.abs(
                (np.cross(Jw, w[:, :, None], axis=1) * u[:, :, None]).sum(1)
                + (w[:, :, None] * np.cross(a[None, :, None], Jv, axis=1)).sum(1)).max(axis=0))
        P, W, qv = np.concatenate(P), np.concatenate(W), np.concatenate(qv)
        del q

        rigid = not free or self._tool_dir_is_free(free)   # may we trust W off the grid?
        reach = (float(np.linalg.norm(P - c, axis=1).max())
                 + float(gx @ half) + 0.5 * curve * drift)
        maps = []
        for f in (probes if rigid else (0.0,)):
            L = float(f) * reach
            v = (P + L * W if L else P) - c
            z = v @ a
            rho = np.linalg.norm(v - z[:, None] * a, axis=-1)
            # d(tip + L * dir)/dq_i <= ||d tip/dq_i|| + L, the direction being a unit vector.
            slack = float((gx + L) @ half) + 0.5 * (curve + L) * drift + 1e-4
            h0 = float(z.min()) - slack
            rows = np.floor(rho / cell).astype(np.int64)
            cols = np.floor((z - h0) / cell).astype(np.int64)
            occ = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=bool)
            occ[rows, cols] = True
            maps.append(dict(L=L, occ=occ, h0=h0, slack=slack, dilated={}))

        qmax = None
        if rigid:
            qmax = (float(np.abs(qv).max()) + float(gq @ half)
                    + 0.5 * (reach + 3.0 * curve) * drift + 1e-4)
        env = dict(key=key, cell=cell, axis=a, point=c.copy(), maps=maps, qmax=qmax, reach=reach)
        self._envelope_cache = env
        return env

    def _tool_dir_is_free(self, free):
        """Do the free joints leave ``M @ tool_dir`` alone? Same standard as :meth:`free_joints`."""
        rng = np.random.default_rng(1)
        q = rng.uniform(self.limits[:, 0], self.limits[:, 1], size=(32, self.n))
        d = self.fk(q)[1] @ self.tool_dir
        for i in free:
            r = q.copy()
            r[:, i] = rng.uniform(self.limits[i, 0], self.limits[i, 1], size=32)
            if np.abs(self.fk(r)[1] @ self.tool_dir - d).max() > 1e-9:
                return False
        return True

    def cannot_reach(self, pos, Rt, pos_tol=0.015, rot_tol=np.radians(20.0)):
        """Which of these poses are PROVABLY out of reach, without solving anything.

        True means no joint angles exist that hold the pose to the tolerances given, so the
        solver can be skipped outright. False means nothing -- most of those still fail once
        solved. Shapes broadcast: ``pos`` is (..., 3) and ``Rt`` (..., 3, 3).
        """
        env = getattr(self, "_envelope_cache", None) or self.envelope()
        a, cell = env["axis"], env["cell"]
        v = np.asarray(pos, dtype=float) - env["point"]
        d = np.asarray(Rt, dtype=float) @ self.tool_dir
        out = np.zeros(np.broadcast_shapes(v.shape[:-1], d.shape[:-1]), dtype=bool)
        for m in env["maps"]:
            y = v + m["L"] * d if m["L"] else v
            h = y @ a
            rho = np.linalg.norm(y - h[..., None] * a, axis=-1)
            # Anything within the tolerance of a reachable point is still a candidate, and an
            # orientation off by rot_tol moves a point L away by twice L sin(rot_tol / 2).
            k = int(np.ceil((pos_tol + 2.0 * m["L"] * np.sin(rot_tol / 2.0) + m["slack"]) / cell))
            ok = m["dilated"].get(k)
            if ok is None:
                occ = m["occ"]
                pad = np.zeros((occ.shape[0] + 2 * k, occ.shape[1] + 2 * k), dtype=bool)
                pad[k:k + occ.shape[0], k:k + occ.shape[1]] = occ
                ok = m["dilated"][k] = _dilate(pad, k)
            i = np.floor(rho / cell).astype(np.int64) + k
            j = np.floor((h - m["h0"]) / cell).astype(np.int64) + k
            inside = (i >= 0) & (i < ok.shape[0]) & (j >= 0) & (j < ok.shape[1])
            miss = ~(inside & ok[np.clip(i, 0, ok.shape[0] - 1), np.clip(j, 0, ok.shape[1] - 1)])
            out |= miss

        if env["qmax"] is not None:
            # |Q(wanted) - Q(held)| <= rho * ||direction error|| + ||position error||.
            u = np.cross(a, v)
            out |= np.abs((d * u).sum(-1)) > (env["qmax"] + pos_tol
                                              + 2.0 * np.sin(rot_tol / 2.0)
                                              * np.linalg.norm(u, axis=-1))
        return out

    # ---- inverse ---------------------------------------------------------------------------
    def ik(self, pos, quat, iters=80, seeds=None, w_rot=1.0, lam=0.05, clamp=0.35):
        """Best joints for a batch of 6-DoF tool poses in the base frame.

        Position is the primary task and orientation the secondary one, solved in the null space
        of the first: the arm goes exactly where it is told, and then uses whatever freedom is
        left over to aim as well as it can. That ordering is not a preference, it is what the
        answer means -- a grasp held 3 cm from the object has failed however well it is aimed,
        while one aimed 10 degrees off usually still closes.

        A single weighted objective was tried first and is the wrong shape for this. Weighted at
        0.25 it converged happily on a real recording to 15.8 mm and 4.2 degrees, and called 49%
        of the frames reachable; the truth was 0.1 mm, 8.8 degrees and 100%. It was not
        under-converged, it was optimising the wrong thing.

        Args:
            pos: (N, 3) tool points.
            quat: (N, 4) as ``(x, y, z, w)``, or (N, 3, 3) already as rotation matrices. A
                placement sweep asks the same orientations thousands of times over and only
                moves the positions, so converting once and passing matrices is worth it.
            w_rot: how hard to chase orientation within the null space. 1.0 is "as hard as
                possible"; lower it only if the secondary task makes the primary jitter.
            iters: gradient steps per seed. Convergence here is fast; the accuracy that matters
                is set by the free-joint sweep afterwards, not by this.
            lam: damping. Keeps the step finite through singularities, at the cost of a little
                accuracy near them.

        Returns:
            ``(q, pos_err, rot_err)`` -- joints (N, n), metres, radians. The two errors are what
            the arm could NOT do at the joints returned, so zero means it got there exactly.
        """
        pos = np.atleast_2d(np.asarray(pos, dtype=float))
        quat = np.asarray(quat, dtype=float)
        Rt = quat if quat.ndim == 3 else R.from_quat(np.atleast_2d(quat)).as_matrix()
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        if seeds is None:
            mid = (lo + hi) / 2.0
            seeds = [mid, lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)]
        eye3 = np.eye(3)
        lam2, eps2 = lam ** 2, 1e-8

        def step(q):
            """One damped-least-squares step. Was written out twice below, identically."""
            tip, M, Jv, Jw = self.jacobian(q)
            ev = (pos - tip)[..., None]
            ew = _rotvec(Rt @ np.swapaxes(M, -1, -2))[..., None]
            JvT, JwT = np.swapaxes(Jv, -1, -2), np.swapaxes(Jw, -1, -2)
            G = Jv @ JvT                     # shared by both dampings of the position pinv
            # Two different dampings on purpose. The STEP is damped so it stays finite
            # through singularities; the PROJECTOR is not, because damping it makes the
            # "null space" overlap the primary task and the two then fight -- measured, that
            # left 6 mm of position error standing at convergence.
            # Every pseudo-inverse here is immediately applied to a vector, so it is never
            # formed: J^T (A^-1 x) is two matrix-vector products where J^T A^-1 then times x
            # is a matrix-matrix product first. Same value, half the work.
            y = JwT @ _chol3(Jw @ JwT + lam2 * eye3, ew)
            # ... and (I - Jv^+ Jv) y without ever building the (N, n, n) projector either.
            null_y = y - JvT @ _chol3(G + eps2 * eye3, Jv @ y)
            dq = (JvT @ _chol3(G + lam2 * eye3, ev) + w_rot * null_y)[..., 0]
            return np.clip(q + np.clip(dq, -clamp, clamp), lo, hi)

        # Every seed at once, as one tall batch. The rows never interact, so this is the same
        # arithmetic in a third of the numpy calls -- and a 249-row batch of 3x3 work is made
        # mostly of call overhead, not of flops.
        N, k = pos.shape[0], len(seeds)
        pos = np.tile(pos, (k, 1))
        Rt = np.tile(Rt, (k, 1, 1))
        q = np.repeat(np.asarray(seeds, dtype=float).reshape(k, self.n), N, axis=0)
        for _ in range(iters):
            q = step(q)
        # Sweep the free joints into the right basin, then let the gradient polish them:
        # the sweep's grid is a few degrees wide, which would otherwise be the accuracy
        # floor of the whole solve.
        q = self._polish_free(q, pos, Rt)
        for _ in range(max(8, iters // 4)):
            q = step(q)
        pe, re = self.error(q, pos, Rt)
        score = (pe + 0.02 * re).reshape(k, N)
        q, pe, re = q.reshape(k, N, self.n), pe.reshape(k, N), re.reshape(k, N)
        best = [q[0], pe[0], re[0], score[0]]
        for j in range(1, k):                      # ties keep the earlier seed, as before
            take = score[j] < best[3]
            best = [np.where(take[:, None], q[j], best[0]), np.where(take, pe[j], best[1]),
                    np.where(take, re[j], best[2]), np.where(take, score[j], best[3])]
        return best[0], best[1], best[2]

    def free_joints(self, atol=1e-6):
        """Joints that move the tool POINT nowhere, whatever they do -- so, pure orientation.

        A wrist roll whose axis passes through the grasp point is the usual one. Found rather
        than declared, by looking for a column of the position Jacobian that stays zero across
        the workspace, so it works for any chain that happens to have one.
        """
        rng = np.random.default_rng(0)
        q = rng.uniform(self.limits[:, 0], self.limits[:, 1], size=(32, self.n))
        _, _, Jv, _ = self.jacobian(q)
        return [i for i in range(self.n) if float(np.abs(Jv[:, :, i]).max()) < atol]

    def _polish_free(self, q, pos, Rt, steps=73):
        """Sweep the free joints exhaustively, keeping whatever aims best.

        Gradient descent will not do this for you. Those joints sit exactly in the null space of
        the primary task, so the step through them is driven only by the orientation residual --
        and on a real recording that converged 14 degrees short of what a sweep finds, because
        the orientation objective is not convex in them. One pass of brute force costs a few
        dozen forward evaluations and closes the gap.
        """
        free = self.free_joints()
        if not free:
            return q
        q = q.copy()
        for i in free:
            # The sweep moves joint i and nothing else, so the walk up to it is identical every
            # time; and past it only the orientation is wanted, which neither the tool point nor
            # the link origins feed. One rotation per trial, not a whole forward kinematics.
            _, M0 = self._walk(q, i)
            trial = q.copy()
            best_q = q[:, i].copy()
            best_re = _rot_err(self._turn_tail(M0, trial, i), Rt)
            for v in np.linspace(self.limits[i, 0], self.limits[i, 1], steps):
                trial[:, i] = v
                re = _rot_err(self._turn_tail(M0, trial, i), Rt)
                take = re < best_re
                best_q = np.where(take, v, best_q)
                best_re = np.where(take, re, best_re)
            q[:, i] = best_q
        return q

    def _turn_tail(self, M, q, start):
        """Orientation from joint ``start`` onwards, given the frame before it."""
        C, S = np.cos(q[..., start:, None]), np.sin(q[..., start:, None])
        for k in range(start, self.n):
            M = _turn(np.einsum("...ik,kj->...ij", M, self.rot[k]),
                      self.axis[k], C[..., k - start, :], S[..., k - start, :])
        return M

    def error(self, q, pos, Rt):
        """How far ``q`` lands from the wanted pose: ``(metres, radians)``."""
        tip, M = self.fk(q)
        pe = np.linalg.norm(tip - pos, axis=-1)
        return pe, _rot_err(M, Rt)

    def reachable(self, pos, quat, pos_tol=0.015, rot_tol=np.radians(20.0)):
        """Boolean mask: can this arm hold each of these poses, within tolerance?"""
        _, pe, re = self.ik(pos, quat)
        return (pe < pos_tol) & (re < rot_tol)
