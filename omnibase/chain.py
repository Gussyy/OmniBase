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


def _spin(axis, q):
    """Rotation by angle(s) ``q`` about cardinal ``axis``, as (..., 3, 3).

    Written out rather than handed to ``Rotation.from_rotvec``: a placement sweep calls this a
    few hundred million times, and the generic path dominates the runtime if you let it.
    """
    q = np.asarray(q, dtype=float)
    c, s = np.cos(q), np.sin(q)
    o, l = np.zeros_like(c), np.ones_like(c)
    rows = ([[l, o, o], [o, c, -s], [o, s, c]] if axis == 0 else
            [[c, o, s], [o, l, o], [-s, o, c]] if axis == 1 else
            [[c, -s, o], [s, c, o], [o, o, l]])
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


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
        p = np.zeros(q.shape[:-1] + (3,))
        M = np.broadcast_to(np.eye(3), q.shape[:-1] + (3, 3)).copy()
        for i in range(self.n):
            p = p + M @ self.origin[i]
            M = M @ self.rot[i] @ _spin(self.axis[i], q[..., i])
        return p + (M @ self.tool), M

    def links(self, q):
        """Joint origins plus the tool point, (..., n+1, 3). For drawing the arm, nothing else."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        p = np.zeros(q.shape[:-1] + (3,))
        M = np.broadcast_to(np.eye(3), q.shape[:-1] + (3, 3)).copy()
        out = []
        for i in range(self.n):
            p = p + M @ self.origin[i]
            M = M @ self.rot[i] @ _spin(self.axis[i], q[..., i])
            out.append(p.copy())
        out.append(p + (M @ self.tool))
        return np.stack(out, axis=-2)

    def jacobian(self, q):
        """``(tip, orientation, Jv, Jw)``. Column i is ``(axis_i x (tip - origin_i), axis_i)``."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        m = q.shape[0]
        p = np.zeros((m, 3))
        M = np.broadcast_to(np.eye(3), (m, 3, 3)).copy()
        axes, orig = [], []
        for i in range(self.n):
            p = p + M @ self.origin[i]
            M = M @ self.rot[i]
            axes.append(M @ np.eye(3)[self.axis[i]])
            orig.append(p.copy())
            M = M @ _spin(self.axis[i], q[..., i])
        tip = p + (M @ self.tool)
        Jv = np.stack([np.cross(axes[i], tip - orig[i]) for i in range(self.n)], axis=-1)
        Jw = np.stack(axes, axis=-1)
        return tip, M, Jv, Jw

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
        eyen = np.eye(self.n)

        def pinv(J, d):
            """J^T (J J^T + d^2 I)^-1, batched: (N, 3, n) -> (N, n, 3)."""
            A = J @ np.swapaxes(J, -1, -2) + (d ** 2) * eye3
            return np.swapaxes(J, -1, -2) @ np.linalg.inv(A)

        best = None
        for seed in seeds:
            q = np.broadcast_to(np.asarray(seed, dtype=float), (pos.shape[0], self.n)).copy()
            for _ in range(iters):
                tip, M, Jv, Jw = self.jacobian(q)
                ev = (pos - tip)[..., None]
                ew = _rotvec(Rt @ np.swapaxes(M, -1, -2))[..., None]
                # Two different dampings on purpose. The STEP is damped so it stays finite
                # through singularities; the PROJECTOR is not, because damping it makes the
                # "null space" overlap the primary task and the two then fight -- measured, that
                # left 6 mm of position error standing at convergence.
                null = eyen - pinv(Jv, 1e-4) @ Jv
                dq = (pinv(Jv, lam) @ ev + w_rot * (null @ (pinv(Jw, lam) @ ew)))[..., 0]
                q = np.clip(q + np.clip(dq, -clamp, clamp), lo, hi)
            # Sweep the free joints into the right basin, then let the gradient polish them:
            # the sweep's grid is a few degrees wide, which would otherwise be the accuracy
            # floor of the whole solve.
            q = self._polish_free(q, pos, Rt)
            for _ in range(max(8, iters // 4)):
                tip, M, Jv, Jw = self.jacobian(q)
                ev = (pos - tip)[..., None]
                ew = _rotvec(Rt @ np.swapaxes(M, -1, -2))[..., None]
                null = eyen - pinv(Jv, 1e-4) @ Jv
                dq = (pinv(Jv, lam) @ ev + w_rot * (null @ (pinv(Jw, lam) @ ew)))[..., 0]
                q = np.clip(q + np.clip(dq, -clamp, clamp), lo, hi)
            pe, re = self.error(q, pos, Rt)
            score = pe + 0.02 * re
            if best is None:
                best = [q, pe, re, score]
            else:
                take = score < best[3]
                best = [np.where(take[:, None], q, best[0]), np.where(take, pe, best[1]),
                        np.where(take, re, best[2]), np.where(take, score, best[3])]
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
        return [i for i in range(self.n)
                if float(np.abs(Jv[:, :, i]).max()) < atol]

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
            best_q, (_, best_re) = q[:, i].copy(), self.error(q, pos, Rt)
            for v in np.linspace(self.limits[i, 0], self.limits[i, 1], steps):
                trial = q.copy()
                trial[:, i] = v
                _, re = self.error(trial, pos, Rt)
                take = re < best_re
                best_q = np.where(take, v, best_q)
                best_re = np.where(take, re, best_re)
            q[:, i] = best_q
        return q

    def error(self, q, pos, Rt):
        """How far ``q`` lands from the wanted pose: ``(metres, radians)``."""
        tip, M = self.fk(q)
        pe = np.linalg.norm(tip - pos, axis=-1)
        re = np.arccos(np.clip((np.einsum("...ij,...ij->...", M, Rt) - 1.0) / 2.0, -1.0, 1.0))
        return pe, re

    def reachable(self, pos, quat, pos_tol=0.015, rot_tol=np.radians(20.0)):
        """Boolean mask: can this arm hold each of these poses, within tolerance?"""
        _, pe, re = self.ik(pos, quat)
        return (pe < pos_tol) & (re < rot_tol)
