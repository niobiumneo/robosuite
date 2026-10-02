#!/usr/bin/env python
"""TIAGo Pro (PAL Robotics xacro) -> MuJoCo menagerie-style package -> robosuite assets.

One pipeline, one source of truth.

  stage 1  offline xacro (ament shim, no ROS) -> URDF -> MuJoCo URDF importer -> mj_saveLastXML
  stage 2  declarative lxml overlay (overlay.py) -> menagerie package  pal_tiago_pro/
  stage 3  package -> robosuite files (robot_right.xml, grippers/pal_pro_gripper.xml), fully flattened

Run with MuJoCo 3.5.0 (asserted: the importer output is version dependent), e.g.

  PYTHONPATH=<mujoco-3.5.0-site> python tools/tiago_pro/convert_tiago_pro.py --pal-root ~/pal-robotics
  ... --check      # regenerate into a temp dir and byte-diff against the committed files

The generated files must never be edited by hand.
"""
import argparse
import copy
import filecmp
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import overlay as ov  # noqa: E402

ASSET_ROOT = os.path.join(REPO, "robosuite", "models", "assets")
TIAGO_DIR = os.path.join(ASSET_ROOT, "robots", "tiago_pro")
PKG_DIR = os.path.join(TIAGO_DIR, ov.PACKAGE_DIRNAME)
GRIPPER_XML = os.path.join(ASSET_ROOT, "grippers", "pal_pro_gripper.xml")
GRIPPER_MESH_DIR = os.path.join(ASSET_ROOT, "grippers", "meshes", "pal_pro_gripper")
DEFAULT_MENAGERIE = "/home/user/google-deepmind/mujoco_menagerie"


# ==================================================================================================
# helpers
# ==================================================================================================
def _require_mujoco():
    import mujoco

    if mujoco.__version__ != ov.MUJOCO_VERSION:
        sys.exit(
            "convert_tiago_pro.py needs mujoco==%s (found %s). Prepend a %s install to PYTHONPATH, e.g.\n"
            "  pip install --target /tmp/mj35 mujoco==%s && PYTHONPATH=/tmp/mj35 python %s ..."
            % (ov.MUJOCO_VERSION, mujoco.__version__, ov.MUJOCO_VERSION, ov.MUJOCO_VERSION, sys.argv[0])
        )
    return mujoco


def git_sha(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def pkg_paths(pal_root):
    paths = {}
    for pkg, (clone, sub) in ov.PAL_PACKAGES.items():
        p = os.path.join(pal_root, clone, sub) if sub else os.path.join(pal_root, clone)
        if not os.path.isdir(p):
            sys.exit("missing PAL package %s at %s" % (pkg, p))
        paths[pkg] = p
    paths["realsense2_description"] = os.path.join(HERE, "stubs", "realsense2_description")
    return paths


def fmt_num(x, nd=8):
    s = ("%.*g" % (nd, x))
    return "0" if s in ("-0", "0") else s


def fmt_vec(v, nd=8):
    return " ".join(fmt_num(float(x), nd) for x in v)


# ==================================================================================================
# stage 1: xacro -> URDF -> compiled MJCF
# ==================================================================================================
def run_xacro(pal_root):
    """Expand tiago_pro.urdf.xacro offline. Returns the URDF text."""
    paths = pkg_paths(pal_root)
    os.environ["TIAGO_PKG_PATHS"] = os.pathsep.join(paths.values())
    sys.path.insert(0, os.path.join(HERE, "shim"))
    import xacro

    top = os.path.join(paths["tiago_pro_description"], ov.URDF_TOP)
    doc = xacro.process_file(top, mappings=dict(ov.XACRO_ARGS))
    return doc.toprettyxml(indent="  ")


def sanitize_urdf(urdf_text, pal_root, work_dir):
    """Strip non-MuJoCo content, rewrite package:// mesh uris to assets/<category>/<name>.stl, copy the meshes.

    Returns (urdf_path, info dict). Meshes are copied flat per category; basenames must be unique overall.
    """
    from lxml import etree

    paths = pkg_paths(pal_root)
    root = etree.fromstring(urdf_text.encode())
    for el in list(root.iter(etree.Comment)):
        el.getparent().remove(el)
    for tag in ("gazebo", "ros2_control", "transmission", "mujoco"):
        for el in root.findall(tag):
            root.remove(el)
    mimic = {}
    for j in root.findall("joint"):
        for el in j.findall("mimic"):
            mimic[j.get("name")] = (el.get("joint"), float(el.get("multiplier", "1")), float(el.get("offset", "0")))
            j.remove(el)
        for tag in ("safety_controller", "calibration"):
            for el in j.findall(tag):
                j.remove(el)
    for l in root.findall("link"):
        for tag in ("kinematic", "gravity"):
            for el in l.findall(tag):
                l.remove(el)
    links, joints = root.findall("link"), root.findall("joint")
    assert len(links) == ov.EXPECTED_URDF_LINKS and len(joints) == ov.EXPECTED_URDF_JOINTS, (
        "unexpected URDF size: %d links / %d joints" % (len(links), len(joints))
    )
    assets = os.path.join(work_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    seen = {}  # basename -> (category, src)
    import re

    for m in root.iter("mesh"):
        mm = re.match(r"package://([^/]+)/+(.*)", m.get("filename"))
        pkg, rel = mm.group(1), mm.group(2)
        cat = None
        for prefix, c in ov.MESH_CATEGORY.get(pkg, {}).items():
            if rel.startswith(prefix + "/"):
                cat = c
        if cat is None:
            # e.g. pal_urdf_utils laser meshes: copied into a quarantine category, geoms dropped in stage 2
            cat = "excluded"
        src = os.path.join(paths[pkg], rel)
        assert os.path.exists(src), src
        base = os.path.basename(rel)
        if base in seen:
            assert seen[base][1] == src, "mesh basename clash %s" % base
        else:
            seen[base] = (cat, src)
            os.makedirs(os.path.join(assets, cat), exist_ok=True)
            shutil.copyfile(src, os.path.join(assets, cat, base))
        m.set("filename", "%s/%s" % (cat, base))
    mj = etree.Element("mujoco")
    comp = etree.SubElement(mj, "compiler")
    for k, v in (("meshdir", "assets"), ("strippath", "false"), ("balanceinertia", "true"), ("autolimits", "true"),
                 ("discardvisual", "false"), ("fusestatic", "false")):
        comp.set(k, v)
    root.insert(0, mj)
    etree.indent(root, space="  ")
    out = os.path.join(work_dir, "tiago_pro.urdf")
    with open(out, "wb") as f:
        f.write(etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True))
    # velocity limits (for the velocity variant) and efforts of every non-fixed joint
    limits = {}
    for j in joints:
        lim = j.find("limit")
        if j.get("type") != "fixed" and lim is not None:
            limits[j.get("name")] = {k: float(lim.get(k)) for k in ("effort", "velocity") if lim.get(k)}
    urdf_mass = sum(float(m.get("value")) for l in links for m in l.findall("inertial/mass"))
    return out, {"links": len(links), "joints": len(joints), "mass": urdf_mass, "limits": limits, "mimic": mimic,
                 "meshes": {b: c for b, (c, _) in seen.items()}}


def stage1(pal_root, out_dir):
    """xacro -> urdf -> compiled MJCF in out_dir/{tiago_pro.urdf, tiago_pro.xml, assets/}. Returns info."""
    mujoco = _require_mujoco()
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    text = run_xacro(pal_root)
    with open(os.path.join(out_dir, "expanded.urdf"), "w") as f:
        f.write(text)
    urdf, info = sanitize_urdf(text, pal_root, out_dir)
    m = mujoco.MjModel.from_xml_path(urdf)
    assert m.neq == 0, "URDF import produced equality constraints (mimic handled by the overlay instead)"
    mujoco.mj_saveLastXML(os.path.join(out_dir, "tiago_pro.xml"), m)
    mass = float(m.body_mass.sum())
    assert abs(mass - info["mass"]) < 1e-6, "mass mismatch URDF %.9f vs MJCF %.9f" % (info["mass"], mass)
    for b in ("arm_right_tool_link", "arm_left_tool_link", "torso_base_link", "base_footprint"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0, b
    info.update(nbody=m.nbody, njnt=m.njnt, ngeom=m.ngeom, nmesh=m.nmesh, mjmass=mass)
    return info


# ==================================================================================================
# stage 2: overlay -> menagerie package
# ==================================================================================================
def read_stl(path):
    import numpy as np

    b = open(path, "rb").read()
    n = int(np.frombuffer(b, dtype="<u4", count=1, offset=80)[0])
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(b, dtype=rec, count=n, offset=84)["v"].reshape(-1, 3).astype(np.float64)


def dedupe_meshes(s1_dir, mesh_files):
    """mesh_files: {mesh_name: 'cat/base.stl'}.  Returns {mesh_name: (canonical file, scale string or None)}.

    Byte-identical files collapse to one file; mirrored copies (X_reflected.stl) become a scaled reference.
    """
    out, by_hash, verts = {}, {}, {}
    for name, f in mesh_files.items():
        path = os.path.join(s1_dir, "assets", f)
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        if h in by_hash:
            out[name] = (by_hash[h], None)
            continue
        key = set(map(tuple, (read_stl(path).round(5) + 0.0).tolist()))
        found = None
        if "_reflected" in f:
            for cf, cv in verts.items():
                for ax in range(3):
                    mirrored = set(tuple(c * (-1.0 if i == ax else 1.0) + 0.0 for i, c in enumerate(v)) for v in cv)
                    if mirrored == key:
                        found = (cf, " ".join("-1" if i == ax else "1" for i in range(3)))
                        break
                if found:
                    break
        if found:
            out[name] = found
            continue
        by_hash[h] = f
        verts[f] = key
        out[name] = (f, None)
    return out


class Pkg:
    """Builds the menagerie package (lxml) from the stage-1 compiled MJCF."""

    def __init__(self, s1_dir, info, shas):
        from lxml import etree

        self.etree = etree
        self.s1 = s1_dir
        self.info = info
        self.shas = shas
        self.src = etree.parse(os.path.join(s1_dir, "tiago_pro.xml")).getroot()
        self.mesh_files = {m.get("name"): m.get("file") for m in self.src.find("asset").findall("mesh")}
        self.mesh_map = dedupe_meshes(s1_dir, self.mesh_files)
        self.used_meshes = []

    # ---- helpers
    def E(self, tag, **attrs):
        el = self.etree.Element(tag)
        for k, v in attrs.items():
            el.set(k, str(v))
        return el

    def sub(self, parent, tag, **attrs):
        el = self.etree.SubElement(parent, tag)
        for k, v in attrs.items():
            el.set(k, str(v))
        return el

    # ---- body tree transformation
    def transform_bodies(self):
        wb = self.wb = self.src.find("worldbody")
        root_body = wb.find("body")
        assert root_body.get("name") == ov.ROOT_BODY
        root_body.set("pos", "0 0 %s" % fmt_num(ov.ROOT_Z))
        fj = self.etree.Element("freejoint")
        fj.set("name", ov.FREEJOINT)
        root_body.insert(0, fj)
        self.joints = {}  # name -> element
        for b in list(root_body.iter("body")):
            self._body(b)
        return wb

    def _body(self, b):
        name = b.get("name")
        keep_col = name in ov.COLLISION_BODIES
        ncol = 0
        for g in list(b.findall("geom")):
            mesh = g.get("mesh")
            is_vis = g.get("group") == "1" and g.get("contype") == "0"
            if mesh in ov.EXCLUDED_MESHES:
                b.remove(g)
                continue
            if is_vis:
                if g.get("size") and g.get("size") == "0.001" and mesh is None and g.get("type") is None:
                    b.remove(g)
                    continue
                for k in ("group", "contype", "conaffinity", "density"):
                    g.attrib.pop(k, None)
                g.set("class", "visual")
            else:
                if (not keep_col) or mesh in ov.DROP_COLLISION_MESHES or (mesh is None and g.get("type") is None):
                    b.remove(g)
                    continue
                g.attrib.pop("rgba", None)
                g.set("class", self._col_class(name))
                ncol += 1
                g.set("name", "%s_collision" % name if ncol == 1 else "%s_collision_%d" % (name, ncol))
            if mesh is not None:
                g.attrib.pop("type", None)
                self.used_meshes.append(mesh)
            # attribute order: class first
            attrs = dict(g.attrib)
            g.attrib.clear()
            for k in ("name", "class", "type", "size", "pos", "quat", "mesh", "rgba"):
                if k in attrs:
                    g.set(k, attrs.pop(k))
            for k, v in attrs.items():
                g.set(k, v)
        for j in b.findall("joint"):
            self._joint(j)
        # keep inertial first
        # frame-only gripper helper bodies keep their inertial (PAL masses)

    def _col_class(self, body):
        if body.startswith("gripper_") and "fingertip" in body:
            return "pad_collision"
        if body.startswith("gripper_") and ("finger" in body):
            return "finger_collision"
        return "collision"

    def _joint(self, j):
        name = j.get("name")
        self.joints[name] = j
        cls = None
        if name.startswith("arm_"):
            cls = "arm"
        elif name.startswith("wheel_"):
            cls = "wheel"
        elif name.startswith("head_"):
            cls = "head"
        elif name == "torso_lift_joint":
            cls = "torso"
        j.attrib.pop("pos", None)
        if cls:
            for k, v in ov.JOINT_CLASSES[cls].items():
                if j.get(k) == v:
                    del j.attrib[k]
                else:
                    assert j.get(k) is None or cls == "torso", (name, k, j.get(k), v)
            j.set("class", cls)
        if name.startswith("gripper_"):
            j.set("actuatorfrcrange", "-%s %s" % (fmt_num(ov.GRIPPER_FINGER_FORCE), fmt_num(ov.GRIPPER_FINGER_FORCE)))
            if name.endswith("_finger_joint"):
                j.set("damping", fmt_num(ov.GRIPPER_FINGER_DAMPING))
            else:
                j.attrib.pop("damping", None)
        attrs = dict(j.attrib)
        j.attrib.clear()
        for k in ("name", "class", "type", "axis", "range", "actuatorfrcrange", "damping", "frictionloss"):
            if k in attrs:
                j.set(k, attrs.pop(k))
        assert not attrs, (name, attrs)

    # ---- additions ---------------------------------------------------------------------------------
    def bodies(self):
        return {b.get("name"): b for b in self.wb.iter("body")}

    def _add_child(self, body, el):
        """Append el after the last non-body child of body (before child bodies)."""
        idx = 0
        for i, c in enumerate(body):
            if c.tag != "body":
                idx = i + 1
        body.insert(idx, el)

    def add_frames(self):
        B = self.bodies()
        for side in ("left", "right"):
            self._add_child(B["arm_%s_tool_link" % side], self.E("site", name="attachment_site_%s" % side,
                                                                  **{"class": "marker"}))
            g = B["gripper_%s_base_link" % side]
            self._add_child(g, self.E("site", name="gripper_%s_ft_frame" % side, **{"class": "marker"}))
            self._add_child(g, self.E("site", name="gripper_%s_grasping_frame_Z" % side, pos=fmt_vec(ov.GRASP_SITE_POS),
                                       **{"class": "marker"}))
        # optical frame (z forward) -> MuJoCo camera frame (z backward) = rotate pi about x
        self._add_child(B[ov.HEAD_CAMERA_FRAME], self.E("camera", name="head_front_camera", fovy=ov.HEAD_CAMERA_FOVY,
                                                         quat="0 1 0 0"))

    def equality(self):
        eq = self.E("equality")
        mimic = self.info["mimic"]
        for side in ("left", "right"):
            g = "gripper_%s_" % side
            for fs, tag in (("right", "r"), ("left", "l")) if ov.GRIPPER_CONNECT else ():
                self.sub(eq, "connect", name="%sconnect_%s" % (g, tag), body1="%sfingertip_%s_link" % (g, fs),
                         body2="%souter_finger_%s_link" % (g, fs), anchor=ov.GRIPPER_CONNECT_ANCHOR[fs if fs == "right" else "left"],
                         active="true", **ov.GRIPPER_EQ)
            for follower, (master, mult, off) in mimic.items():
                if follower.startswith(g):
                    assert off == 0.0 and master == g + "finger_joint", (follower, master)
                    self.sub(eq, "joint", name="%s_mimic" % follower, joint1=master, joint2=follower,
                             polycoef="0 %s 0 0 0" % fmt_num(1.0 / mult), **ov.GRIPPER_EQ)
        return eq

    def contacts(self):
        ct = self.E("contact")
        pairs = list(ov.CONTACT_EXCLUDES)
        for side in ("left", "right"):
            for a, b in ov.GRIPPER_CONTACT_EXCLUDES:
                for fs in ("left", "right"):
                    pairs.append(("gripper_%s_%s" % (side, a.format(s=fs)), "gripper_%s_%s" % (side, b.format(s=fs))))
        seen = set()
        B = self.bodies()
        for a, b in pairs:
            a = a if a in B else "gripper_" + a
            if a in B and b in B and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                self.sub(ct, "exclude", body1=a, body2=b)
        return ct

    def sensors(self):
        se = self.E("sensor")
        self.sub(se, "force", name="gripper_right_force", site="gripper_right_ft_frame")
        self.sub(se, "torque", name="gripper_right_torque", site="gripper_right_ft_frame")
        return se

    def defaults(self):
        d = self.E("default")
        self.sub(d, "mesh", maxhullvert=ov.MESH_MAXHULLVERT)
        vis = self.sub(d, "default", **{"class": "visual"})
        self.sub(vis, "geom", type="mesh", group="2", contype="0", conaffinity="0", density="0")
        col = self.sub(d, "default", **{"class": "collision"})
        self.sub(col, "geom", type="mesh", group="3")
        fc = self.sub(col, "default", **{"class": "finger_collision"})
        self.sub(fc, "geom", **ov.FINGER_GEOM)
        pc = self.sub(col, "default", **{"class": "pad_collision"})
        self.sub(pc, "geom", **ov.PAD_GEOM)
        for cls, attrs in ov.JOINT_CLASSES.items():
            c = self.sub(d, "default", **{"class": cls})
            self.sub(c, "joint", **attrs)
        mk = self.sub(d, "default", **{"class": "marker"})
        self.sub(mk, "site", size="0.005", group="4", rgba="1 0.3 0.3 0.6")
        return d

    def assets(self):
        a = self.E("asset")
        used = set(self.used_meshes)
        files = {}
        for name in self.mesh_files:
            if name not in used:
                continue
            f, scale = self.mesh_map[name]
            files[f] = True
            attrs = {"name": name, "file": f}
            if scale:
                attrs["scale"] = scale
            self.sub(a, "mesh", **attrs)
        self.package_meshes = sorted(files)
        return a

    def main_xml(self):
        root = self.E("mujoco", model=ov.MODEL_NAME)
        c = self.sub(root, "compiler", angle="radian", meshdir="assets", autolimits="true")
        self.sub(root, "option", integrator="implicitfast")
        wb = self.transform_bodies()
        self.add_frames()
        root.append(self.defaults())
        root.append(self.assets())
        root.append(wb)
        root.append(self.contacts())
        root.append(self.equality())
        root.append(self.sensors())
        root.insert(0, self.etree.Comment(
            " Generated by tools/tiago_pro/convert_tiago_pro.py (do not edit). Derived from PAL Robotics TIAGo Pro "
            "xacro (Apache-2.0), modified. See README.md. "))
        return root

    # ---- variants / scenes
    def joint_limits(self, jname):
        return self.info["limits"][jname]

    def actuators(self, variant, model_names):
        """Actuator element for a variant. Order: wheels, torso, head, left arm, left gripper, right arm, right gripper."""
        ac = self.E("actuator")
        J = self.joints
        for w in ("front_right", "front_left", "rear_right", "rear_left"):
            self.sub(ac, "velocity", name="wheel_%s_joint_velocity" % w, joint="wheel_%s_joint" % w,
                     ctrlrange="-%s %s" % (ov.WHEEL_CTRLRANGE, ov.WHEEL_CTRLRANGE), kv=ov.WHEEL_KV)
        r = J["torso_lift_joint"].get("range")
        self.sub(ac, "position", name="torso_lift_joint_position", joint="torso_lift_joint", ctrlrange=r, kp=ov.TORSO_KP)
        for h in (1, 2):
            self.sub(ac, "position", name="head_%d_joint_position" % h, joint="head_%d_joint" % h,
                     ctrlrange=J["head_%d_joint" % h].get("range"), kp=ov.HEAD_KP)
        for side in ("left", "right"):
            for i in range(1, 8):
                jn = "arm_%s_%d_joint" % (side, i)
                lim = self.joint_limits(jn)
                if variant == "position":
                    self.sub(ac, "position", name=jn.replace("_joint", "_joint_position"), joint=jn,
                             ctrlrange=J[jn].get("range"), kp=ov.ARM_KP[i - 1], kv=ov.ARM_KV[i - 1])
                elif variant == "velocity":
                    v = fmt_num(lim["velocity"])
                    self.sub(ac, "velocity", name=jn.replace("_joint", "_joint_velocity"), joint=jn,
                             ctrlrange="-%s %s" % (v, v), kv=ov.ARM_VELOCITY_KV)
                else:
                    e = fmt_num(lim["effort"])
                    self.sub(ac, "motor", name=jn.replace("_joint", "_joint_torque"), joint=jn,
                             ctrlrange="-%s %s" % (e, e))
            self.sub(ac, "position", name="gripper_%s_finger_joint_position" % side, joint="gripper_%s_finger_joint" % side,
                     ctrlrange="%s %s" % ov.GRIPPER_FINGER_RANGE, kp=ov.GRIPPER_KP)
        return ac

    def home_keyframe(self, variant, pkg_dir):
        """<keyframe> with the home pose; computed with mujoco from the already written files."""
        import mujoco
        import numpy as np

        m = mujoco.MjModel.from_xml_path(os.path.join(pkg_dir, "tiago_pro_%s.xml" % variant))
        d = mujoco.MjData(m)
        q = m.qpos0.copy()

        def setj(name, val):
            q[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = val

        q[:3] = (0, 0, ov.ROOT_Z)
        q[3:7] = (1, 0, 0, 0)
        setj("torso_lift_joint", ov.HOME_TORSO)
        setj("head_1_joint", ov.HOME_HEAD[0])
        setj("head_2_joint", ov.HOME_HEAD[1])
        for side in ("left", "right"):
            for i, v in enumerate(ov.HOME_ARM[side]):
                setj("arm_%s_%d_joint" % (side, i + 1), v)
            setj("gripper_%s_finger_joint" % side, ov.GRIPPER_OPEN)
            for f, (master, mult, off) in self.info["mimic"].items():
                if f.startswith("gripper_%s_" % side):
                    setj(f, mult * ov.GRIPPER_OPEN + off)
        ctrl = np.zeros(m.nu)
        for u in range(m.nu):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[u, 0])
            if m.actuator_biastype[u] == 1 and m.actuator_biasprm[u, 1] != 0:  # position actuator
                ctrl[u] = q[m.jnt_qposadr[m.actuator_trnid[u, 0]]]
        if variant == "motor":  # gravity compensation at home so the arm holds the pose
            d.qpos[:] = q
            d.qvel[:] = 0
            d.qacc[:] = 0
            mujoco.mj_inverse(m, d)
            for u in range(m.nu):
                if m.actuator_biastype[u] == 0 and m.actuator_gaintype[u] == 0 and m.actuator_dyntype[u] == 0 \
                        and m.actuator_trntype[u] == 0:
                    ctrl[u] = float(np.clip(d.qfrc_inverse[m.jnt_dofadr[m.actuator_trnid[u, 0]]],
                                            m.actuator_ctrlrange[u, 0], m.actuator_ctrlrange[u, 1]))
        kf = self.E("keyframe")
        self.sub(kf, "key", name="home", qpos=fmt_vec(q, 8), ctrl=fmt_vec(ctrl.round(4), 8))
        return kf

    def variant_xml(self, variant, pkg_dir, with_key=True):
        root = self.E("mujoco", model=ov.MODEL_NAME)
        self.sub(root, "include", file="tiago_pro.xml")
        root.append(self.actuators(variant, None))
        if with_key:
            root.append(self.home_keyframe(variant, pkg_dir))
        return root

    def scene_xml(self, variant):
        etree = self.etree
        root = self.E("mujoco", model="tiago_pro %s scene" % variant)
        self.sub(root, "include", file="tiago_pro_%s.xml" % variant)
        vis = self.sub(root, "visual")
        self.sub(vis, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
        self.sub(vis, "rgba", haze="0.15 0.25 0.35 1")
        self.sub(vis, "global", azimuth="150", elevation="-20", realtime="1")
        a = self.sub(root, "asset")
        self.sub(a, "texture", type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
        self.sub(a, "texture", type="2d", name="groundplane", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4",
                 rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
        self.sub(a, "material", name="groundplane", texture="groundplane", texuniform="true", texrepeat="5 5",
                 reflectance="0.2")
        wb = self.sub(root, "worldbody")
        self.sub(wb, "light", name="spotlight", mode="targetbody", target=ov.ROOT_BODY, pos="1 0 10")
        self.sub(wb, "geom", name="floor", size="0 0 0.05", type="plane", material="groundplane", contype="1",
                 conaffinity="1")
        return root


def write_xml(path, root, formatter):
    from lxml import etree

    raw = etree.tostring(root, pretty_print=True, encoding="unicode")
    if formatter is not None:
        raw = formatter(raw)
    with open(path, "w") as f:
        f.write(raw if raw.endswith("\n") else raw + "\n")


def load_formatter(path):
    if not path or not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("format_xml", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.format_xml


def stage2(pal_root, s1_dir, info, pkg_dir, formatter):
    """Write the menagerie package (xml + assets); LICENSE/README/CHANGELOG are written by write_docs."""
    shas = {c: git_sha(os.path.join(pal_root, c)) for c in ov.GIT_CLONES}
    pk = Pkg(s1_dir, info, shas)
    os.makedirs(pkg_dir, exist_ok=True)
    main = pk.main_xml()
    write_xml(os.path.join(pkg_dir, "tiago_pro.xml"), main, formatter)
    ad = os.path.join(pkg_dir, "assets")
    if os.path.exists(ad):
        shutil.rmtree(ad)
    for f in pk.package_meshes:
        os.makedirs(os.path.dirname(os.path.join(ad, f)), exist_ok=True)
        shutil.copyfile(os.path.join(s1_dir, "assets", f), os.path.join(ad, f))
    for v in ("position", "velocity", "motor"):
        write_xml(os.path.join(pkg_dir, "tiago_pro_%s.xml" % v), pk.variant_xml(v, pkg_dir, with_key=False), formatter)
        write_xml(os.path.join(pkg_dir, "tiago_pro_%s.xml" % v), pk.variant_xml(v, pkg_dir), formatter)
        write_xml(os.path.join(pkg_dir, "scene_%s.xml" % v), pk.scene_xml(v), formatter)
    return pk


# ==================================================================================================
# stage 3: package -> robosuite files
# ==================================================================================================
ROBOSUITE_GROUP = {"2": "1", "3": "0"}  # package visual/collision groups -> robosuite's (vis=1, col=0)


def collect_defaults(default_el):
    """{class name: {tag: attrs}} with parent inheritance; the unnamed top level default is 'main'."""
    out = {}

    def rec(el, inherited, name):
        cur = {t: dict(a) for t, a in inherited.items()}
        for child in el:
            if child.tag != "default" and isinstance(child.tag, str):
                cur.setdefault(child.tag, {}).update(dict(child.attrib))
        out[name] = cur
        for ch in el.findall("default"):
            rec(ch, cur, ch.get("class"))

    rec(default_el, {}, "main")
    return out


def flatten_defaults(root):
    """Resolve every default class inline; afterwards no <default>, class= or childclass= remain."""
    d = root.find("default")
    if d is None:
        return
    cls = collect_defaults(d)

    def walk(el, childclass):
        if isinstance(el.tag, str):
            if el.tag == "body" and el.get("childclass"):
                childclass = el.attrib.pop("childclass")
            if el.tag in ("geom", "joint", "site", "camera", "mesh", "motor", "position", "velocity"):
                c = el.attrib.pop("class", None) or childclass or "main"
                for k, v in cls[c].get(el.tag, {}).items():
                    if k not in el.attrib:
                        el.set(k, v)
        for ch in el:
            walk(ch, childclass)

    for ch in root:
        if ch is not d:
            walk(ch, None)
    root.remove(d)
    for el in root.iter():
        assert el.get("class") is None and el.get("childclass") is None, "unflattened class on %s" % el.tag


def quat_mul(a, b):
    import mujoco
    import numpy as np

    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def eye_in_hand_quat():
    """Camera frame (looks along -z) with its viewing direction = tool +z tilted toward the tool axis."""
    import numpy as np

    flip = np.array([0.0, 1.0, 0.0, 0.0])  # looking along +z of the tool frame
    th = np.deg2rad(ov.EYE_IN_HAND_TILT_DEG)
    ry = np.array([np.cos(th / 2), 0.0, np.sin(th / 2), 0.0])
    return quat_mul(ry, flip)


def strip_prefix(name, prefix):
    return name[len(prefix):] if name.startswith(prefix) else name


class Robosuite:
    """Derives robot_right.xml and pal_pro_gripper.xml from the written package."""

    FROZEN_PREFIXES = ("wheel_", "torso_lift_joint", "head_", "arm_left_", "gripper_left_")

    def __init__(self, pkg_dir, info):
        from lxml import etree

        self.etree = etree
        self.pkg_dir = pkg_dir
        self.info = info
        self.text = open(os.path.join(pkg_dir, "tiago_pro.xml"), "rb").read()

    @property
    def tree(self):  # fresh copy: every derivation mutates its tree
        return self.etree.fromstring(self.text)

    def E(self, tag, **attrs):
        el = self.etree.Element(tag)
        for k, v in attrs.items():
            el.set(k, str(v))
        return el

    def sub(self, parent, tag, **attrs):
        el = self.etree.SubElement(parent, tag)
        for k, v in attrs.items():
            el.set(k, str(v))
        return el

    def mesh_decl(self, root):
        return {m.get("name"): m for m in root.find("asset").findall("mesh")}

    # ---- forward kinematics of the package at the frozen configuration
    def bake_poses(self):
        import mujoco
        import numpy as np

        m = mujoco.MjModel.from_xml_path(os.path.join(self.pkg_dir, "tiago_pro.xml"))
        d = mujoco.MjData(m)
        q = m.qpos0.copy()

        def setj(name, val):
            q[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = val

        setj("torso_lift_joint", ov.PARK_TORSO)
        setj("head_1_joint", ov.PARK_HEAD[0])
        setj("head_2_joint", ov.PARK_HEAD[1])
        for i, v in enumerate(ov.LEFT_PARK):
            setj("arm_left_%d_joint" % (i + 1), v)
        setj("gripper_left_finger_joint", ov.PARK_LEFT_GRIPPER)
        for f, (master, mult, off) in self.info["mimic"].items():
            if f.startswith("gripper_left_"):
                setj(f, mult * ov.PARK_LEFT_GRIPPER + off)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        poses = {}
        for b in range(1, m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            poses[name] = (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy(), m.body_parentid[b])
        names = {b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)}
        return poses, names, q

    def _is_frozen(self, jname):
        return jname.startswith(self.FROZEN_PREFIXES)

    def robot_right(self, out_path, formatter):
        import mujoco
        import numpy as np

        root = self.tree
        poses, names, _ = self.bake_poses()
        wb = root.find("worldbody")
        base = wb.find("body")
        assert base.get("name") == ov.ROOT_BODY
        # -- bake frozen joints into body pos/quat
        for body in list(wb.iter("body")):
            joints = body.findall("joint") + body.findall("freejoint")
            frozen = [j for j in joints if j.tag == "freejoint" or self._is_frozen(j.get("name"))]
            for j in frozen:
                body.remove(j)
            if frozen and body is not base:
                pw, Rw, par = poses[body.get("name")]
                pp, Rp, _ = poses[names[par]]
                loc_p = Rp.T @ (pw - pp)
                loc_R = Rp.T @ Rw
                qq = np.zeros(4)
                mujoco.mju_mat2Quat(qq, loc_R.reshape(-1))
                body.set("pos", fmt_vec(loc_p, 10))
                body.set("quat", fmt_vec(qq, 10))
        base.attrib.pop("pos", None)  # robot root sits on the floor (z=0) inside the robosuite `base` body
        # -- right hand: replace the right gripper subtree by an empty `right_hand` body at the tool link
        tool = [b for b in wb.iter("body") if b.get("name") == "arm_right_tool_link"][0]
        for ch in list(tool):
            if ch.tag == "body":
                tool.remove(ch)
            elif ch.tag in ("site", "camera"):
                tool.remove(ch)
        tool.set("name", "right_hand")
        assert tool.get("quat") is None, "tool link has a rotation; right_hand quat must be taken from the model"
        tool.set("quat", "1 0 0 0")
        self.sub(tool, "camera", mode="fixed", name="eye_in_hand", pos=fmt_vec(ov.EYE_IN_HAND_POS), quat=fmt_vec(eye_in_hand_quat(), 8),
                 fovy=ov.EYE_IN_HAND_FOVY)
        # -- cameras / sites coming from the package: keep only the head camera as `robotview`
        for el in list(wb.iter("site")):
            el.getparent().remove(el)
        for cam in list(wb.iter("camera")):
            if cam.get("name") == "head_front_camera":
                cam.set("name", "robotview")
            elif cam.get("name") != "eye_in_hand":
                cam.getparent().remove(cam)
        arm_mount = [b for b in wb.iter("body") if b.get("name") == "torso_arm_right_link"][0]
        # `right_center` is the OSC controller's base frame (input_ref_frame "base"): it must be aligned with the ROBOT
        # base frame (x forward, y left, z up), not with the (rotated) PAL arm mount frame it sits on
        R_mount = poses["torso_arm_right_link"][1]
        q_site = np.zeros(4)
        mujoco.mju_mat2Quat(q_site, R_mount.T.reshape(-1))
        self.sub(arm_mount, "site", name="right_center", pos="0 0 0", quat=fmt_vec(q_site, 10), size="0.01", rgba="1 0.3 0.3 1",
                 group="2")
        # -- drop unused / foreign sections
        for tag in ("equality", "sensor", "keyframe", "option", "actuator"):
            for el in root.findall(tag):
                root.remove(el)
        flatten_defaults(root)
        for g in root.iter("geom"):
            if g.get("group") in ROBOSUITE_GROUP:
                g.set("group", ROBOSUITE_GROUP[g.get("group")])
        # -- joints: only the 7 right arm joints remain; add torque motors with the URDF efforts
        jn = [j.get("name") for j in wb.iter("joint")]
        assert jn == ["arm_right_%d_joint" % i for i in range(1, 8)], jn
        act = self.E("actuator")
        for i in range(1, 8):
            e = fmt_num(self.info["limits"]["arm_right_%d_joint" % i]["effort"])
            self.sub(act, "motor", name="arm_right_%d_joint_torque" % i, joint="arm_right_%d_joint" % i, ctrllimited="true",
                     ctrlrange="-%s %s" % (e, e))
        root.insert(list(root).index(root.find("asset")), act)
        # -- contact excludes: keep those whose bodies still exist
        alive = {b.get("name") for b in wb.iter("body")}
        ct = root.find("contact")
        for ex in list(ct):
            if ex.get("body1") not in alive or ex.get("body2") not in alive:
                ct.remove(ex)
        # -- assets: robosuite resolves `file` relative to the xml, so reference the package assets in place
        used = {g.get("mesh") for g in wb.iter("geom") if g.get("mesh")}
        for m_ in list(root.find("asset")):
            if m_.get("name") not in used:
                root.find("asset").remove(m_)
            else:
                m_.set("file", "%s/assets/%s" % (ov.PACKAGE_DIRNAME, m_.get("file")))
        comp = root.find("compiler")
        comp.attrib.pop("meshdir", None)
        # -- wrap in the robosuite `base` body
        wrapper = self.E("body", name="base", pos="0 0 0")
        self.sub(wrapper, "inertial", pos="0 0 0", mass="0", diaginertia="0 0 0")
        base.getparent().remove(base)
        wrapper.append(base)
        wb.append(wrapper)
        root.set("model", ov.ROBOT_RIGHT_MODEL)
        for c in list(root):
            if c.tag is self.etree.Comment:
                root.remove(c)
        root.insert(0, self.etree.Comment(
            " Generated by tools/tiago_pro/convert_tiago_pro.py (do not edit); derived from PAL Robotics TIAGo Pro (Apache-2.0), "
            "modified: right arm only, everything else frozen/baked. "))
        write_xml(out_path, root, formatter)

    # ---------------------------------------------------------------------------------------------
    def gripper(self, out_path, mesh_dir, formatter):
        import shutil

        root = self.tree
        wb = root.find("worldbody")
        gb = [b for b in wb.iter("body") if b.get("name") == "gripper_right_base_link"][0]
        P = "gripper_right_"
        # re-root
        newroot = self.E("mujoco", model="pal_pro_gripper")
        newroot.append(self.etree.Comment(
            " Generated by tools/tiago_pro/convert_tiago_pro.py (do not edit); derived from PAL Robotics pal_pro_gripper_description "
            "(Apache-2.0), modified. Joint order: finger_joint (actuated, 0=closed .. 0.07=open) + mimic joints. "))
        flat_src = self.etree.Element("mujoco")
        flat_src.append(root.find("default"))
        asset_src = root.find("asset")
        # collect only the gripper meshes
        gmesh = {g.get("mesh") for g in gb.iter("geom") if g.get("mesh")}
        a = self.sub(newroot, "asset")
        for m_ in asset_src.findall("mesh"):
            if m_.get("name") in gmesh:
                attrs = dict(m_.attrib)
                attrs["maxhullvert"] = str(ov.MESH_MAXHULLVERT)
                fname = os.path.basename(attrs["file"])
                attrs["file"] = "meshes/pal_pro_gripper/%s" % fname
                self.sub(a, "mesh", **attrs)
        # actuator
        ac = self.sub(newroot, "actuator")
        self.sub(ac, "position", name="finger_joint_position", ctrllimited="true", ctrlrange="%s %s" % ov.GRIPPER_FINGER_RANGE,
                 joint="finger_joint", kp=ov.GRIPPER_KP, forcelimited="true",
                 forcerange="-%s %s" % (fmt_num(ov.GRIPPER_FINGER_FORCE), fmt_num(ov.GRIPPER_FINGER_FORCE)))
        # worldbody
        nwb = self.sub(newroot, "worldbody")
        body = copy.deepcopy(gb)
        # remove package frames (grasping link, sites) and add robosuite sites/eef
        for el in list(body.iter("site")):
            el.getparent().remove(el)
        for b in list(body.iter("body")):
            if b.get("name") == P + "grasping_link":
                b.getparent().remove(b)
        # rename
        for el in body.iter():
            if not isinstance(el.tag, str):
                continue
            for k in ("name", "joint"):
                v = el.get(k)
                if v and v.startswith(P) and el.tag in ("body", "joint"):
                    el.set(k, v[len(P):])
        body.set("name", "right_gripper")
        body.set("pos", "0 0 0")
        body.set("quat", "1 0 0 0")
        for g in body.iter("geom"):
            n = g.get("name")
            if n:
                n = n.replace(P, "")
                n = n.replace("base_link_collision", "hand_collision").replace("_link_collision", "_collision")
                g.set("name", n)
        # sites + eef
        ft = self.E("site", name="ft_frame", pos="0 0 0", size="0.01 0.01 0.01", rgba="1 0 0 1", type="sphere", group="1")
        idx = 0
        for i, c in enumerate(body):
            if c.tag == "inertial":
                idx = i + 1
        body.insert(idx, ft)
        eef = self.E("body", name="eef", pos=fmt_vec(ov.GRASP_SITE_POS), quat="1 0 0 0")
        self.sub(eef, "site", name="grip_site", pos="0 0 0", size="0.01 0.01 0.01", rgba="1 0 0 0.5", type="sphere", group="1")
        self.sub(eef, "site", name="ee_x", pos="0.1 0 0", size="0.005 .1", quat="0.707105 0 0.707108 0", rgba="1 0 0 0",
                 type="cylinder", group="1")
        self.sub(eef, "site", name="ee_y", pos="0 0.1 0", size="0.005 .1", quat="0.707105 0.707108 0 0", rgba="0 1 0 0",
                 type="cylinder", group="1")
        self.sub(eef, "site", name="ee_z", pos="0 0 0.1", size="0.005 .1", quat="1 0 0 0", rgba="0 0 1 0", type="cylinder",
                 group="1")
        self.sub(eef, "site", name="grip_site_cylinder", pos="0 0 0", size="0.005 10", rgba="0 1 0 0.3", type="cylinder", group="1")
        idx = 0
        for i, c in enumerate(body):
            if c.tag != "body":
                idx = i + 1
        body.insert(idx, eef)
        nwb.append(body)
        # equality (mimic joints of the right gripper), contact excludes, sensors
        eq = self.sub(newroot, "equality")
        for follower, (master, mult, off) in self.info["mimic"].items():
            if follower.startswith(P):
                self.sub(eq, "joint", name=strip_prefix(follower, P) + "_mimic", joint1=strip_prefix(master, P),
                         joint2=strip_prefix(follower, P), polycoef="0 %s 0 0 0" % fmt_num(1.0 / mult), **ov.GRIPPER_EQ)
        ct = self.sub(newroot, "contact")
        alive = {b.get("name") for b in nwb.iter("body")}
        seen = set()
        for a_, b_ in ov.GRIPPER_CONTACT_EXCLUDES:
            for fs in ("left", "right"):
                n1 = a_.format(s=fs)
                n2 = b_.format(s=fs)
                n1 = "right_gripper" if n1 == "base_link" else n1
                if n1 in alive and n2 in alive and frozenset((n1, n2)) not in seen:
                    seen.add(frozenset((n1, n2)))
                    self.sub(ct, "exclude", body1=n1, body2=n2)
        se = self.sub(newroot, "sensor")
        self.sub(se, "force", name="force_ee", site="ft_frame")
        self.sub(se, "torque", name="torque_ee", site="ft_frame")
        # flatten with the package defaults
        newroot.insert(1, flat_src.find("default"))
        flatten_defaults(newroot)
        for g in newroot.iter("geom"):
            if g.get("group") in ROBOSUITE_GROUP:
                g.set("group", ROBOSUITE_GROUP[g.get("group")])
        # joint equality names must not clash; mimic joints keep range/actuatorfrcrange from the package
        os.makedirs(mesh_dir, exist_ok=True)
        for f in os.listdir(mesh_dir):
            os.remove(os.path.join(mesh_dir, f))
        for m_ in newroot.find("asset"):
            src = None
            for name, mm in self.mesh_decl(root).items():
                if name == m_.get("name"):
                    src = os.path.join(self.pkg_dir, "assets", mm.get("file"))
            shutil.copyfile(src, os.path.join(mesh_dir, os.path.basename(m_.get("file"))))
        write_xml(out_path, newroot, formatter)


# ==================================================================================================
# provenance lock + documentation
# ==================================================================================================
LOCK_PATH = os.path.join(HERE, "sources.lock")


def compute_lock(pal_root):
    import lxml.etree
    import xacro

    lines = ["# sources.lock: inputs of the generated TIAGo Pro assets (written by convert_tiago_pro.py --update-lock)",
             "# <name> <git sha> <url>   |   tool <name> <version>"]
    for c in ov.GIT_CLONES:
        url = subprocess.check_output(["git", "-C", os.path.join(pal_root, c), "remote", "get-url", "origin"], text=True).strip()
        lines.append("%s %s %s" % (c, git_sha(os.path.join(pal_root, c)), url))
    import mujoco

    lines.append("tool mujoco %s" % mujoco.__version__)
    lines.append("tool xacro %s" % getattr(xacro, "__version__", "2.1.1"))
    lines.append("tool lxml %s" % ".".join(map(str, lxml.etree.LXML_VERSION)))
    return "\n".join(lines) + "\n"


def read_lock():
    shas = {}
    for line in open(LOCK_PATH):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        if t[0] != "tool":
            shas[t[0]] = (t[1], t[2])
    return shas


LICENSE_MD5 = "86d3f3a95c324c9479bd8986968f4327"

README_PKG = """## TIAGo Pro Description (MJCF)

> [!IMPORTANT]
> Requires MuJoCo 3.3.0 or later (loaded and stepped with 3.3.0, 3.5.0 and 3.14.0).

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for a full history of changes.

### Overview

This package contains a simplified robot description (MJCF) of the [TIAGo Pro](https://pal-robotics.com/robots/tiago-pro/)
robot with omnidirectional base, two 7-DoF arms and two PAL Pro grippers, by
[PAL Robotics](https://pal-robotics.com/). It is derived from the publicly available xacro/URDF descriptions listed
below. The model is generated by `tools/tiago_pro/convert_tiago_pro.py` of the robosuite fork this package ships in;
the files in this directory are not meant to be edited by hand.

Scenes: `scene_position.xml`, `scene_velocity.xml` and `scene_motor.xml` select the actuator variant
(`tiago_pro_position.xml`, `tiago_pro_velocity.xml`, `tiago_pro_motor.xml`; all include `tiago_pro.xml`). Every variant
has a `home` keyframe. Joint names are the ones of the PAL URDF (`arm_{left,right}_{1..7}_joint`, `torso_lift_joint`,
`head_{1,2}_joint`, `wheel_*_joint`, `gripper_{left,right}_finger_joint`).

| Variant | Wheels | Torso / head | Arms | Grippers |
|---|---|---|---|---|
| position | velocity (kv 500) | position | position (PAL kp/kv) | position (kp 100, PAL) |
| velocity | velocity (kv 500) | position | velocity (kv 100) | position |
| motor | velocity (kv 500) | position | torque, ctrlrange = URDF efforts (43/26 N m) | position |

The right gripper carries a `force`/`torque` sensor pair (`gripper_right_force`, `gripper_right_torque`) on the site
`gripper_right_ft_frame`, which sits on the gripper base body (proximal of all fingers), i.e. it measures the full wrist
load of the gripper.

### URDF -> MJCF derivation steps

 1. `xacro robots/tiago_pro.urdf.xacro` expanded offline (no ROS; `$(find pkg)` resolved by a small shim) with
    `camera_model:=realsense-d435`, `tool_changer_{left,right}:=True`, `end_effector_{left,right}:=pal-pro-gripper`,
    `ft_sensor_*:=no-ft-sensor`, spherical wrist, `limits_v2:=False`, no teleop arms, no calibration tool.
 2. Removed `<gazebo>`, `<ros2_control>`, `<transmission>`, `<safety_controller>` and `<mimic>` elements; rewrote
    `package://` mesh uris to `assets/<category>/<name>.stl`.
 3. Added `<mujoco><compiler meshdir="assets" strippath="false" balanceinertia="true" autolimits="true"
    discardvisual="false" fusestatic="false"/></mujoco>` and compiled with the MuJoCo 3.5.0 URDF importer
    (`fusestatic="false"` keeps the tool, grasp and camera frames), saved with `mj_saveLastXML`.
 4. Overlay (all declarative, `tools/tiago_pro/overlay.py`): free joint `reference` on `base_footprint` (2 mm above the
    floor at z=0), `visual` (group 2, no contact) and `collision` (group 3) default classes with
    `maxhullvert="64"`, joint classes (`arm`, `wheel`, `head`, `torso`), collision geoms kept only for the base (3 boxes),
    wheels (cylinders), torso, head, arms and grippers, contact excludes for always-touching neighbours, sites, a head
    camera, force/torque sensors, actuator variants, `home` keyframe, scenes.
 5. Gripper: PAL's `<mimic>` couplings are expressed as `<equality><joint>` constraints (the MuJoCo URDF importer
    ignores `mimic` in 3.5.0), the finger joint uses PAL's MuJoCo recipe (`actuatorfrcrange` 8, damping 1.5, position
    actuator kp 100). Finger geoms use the contact parameters of robosuite's Panda gripper (friction, condim, solref).
 6. Identical STL files are stored once and mirrored copies (`*_reflected`) are `scale`d references of one file.
 7. Formatted with the menagerie `format_xml.py`.

### Known limitations

- The omnidirectional wheels are isotropic cylinders: the base cannot strafe like the real robot.
- PAL's closed-loop `connect` constraints of the gripper four-bar are not emitted (their anchors fight the mimic
  equalities); the kinematics follow the URDF `mimic` joints.
- `realsense2_description` is not part of the PAL repositories and is replaced by a hand-written stand-in with the
  usual frame names. Only the head camera frames (`head_front_camera_*`) depend on it; their offsets may be off by ~2 cm.
- Laser scanner meshes (`sick_tim551`, from `pal_urdf_utils`, different license) are not redistributed; the laser
  bodies exist as massive frames without geometry.
- No thumbnail image (needs a GL context).

### Derived from

__DERIVED__

### License

This model is released under an [Apache-2.0 License](LICENSE).
"""

CHANGELOG_PKG = """# Changelog – TIAGo Pro Description

All notable changes to this model will be documented in this file.

## [2026-10-02]
- Initial release.
"""

README_TIAGO = """# TIAGo Pro assets

Everything in this directory is **generated** by `tools/tiago_pro/convert_tiago_pro.py` from the PAL Robotics TIAGo Pro
description (Apache-2.0), pinned in `tools/tiago_pro/sources.lock`. Do not edit by hand; edit
`tools/tiago_pro/overlay.py` and re-run the converter.

- `pal_tiago_pro/`: self-contained MuJoCo-menagerie-style package (full robot, position/velocity/motor variants,
  scenes, meshes). It can be copied verbatim into a menagerie checkout.
- `robot_right.xml`: robosuite robot model derived from the package: only the 7 right arm joints are free
  (torque motors); free joint, wheels, torso, head, left arm and left gripper are frozen by baking their pose into the
  body transforms (masses are kept). The right gripper is mounted by robosuite (`right_hand` is an empty body).
  Fully flattened (no `<default>`/`class`), mesh files are the package's (`pal_tiago_pro/assets/...`).
- `../../grippers/pal_pro_gripper.xml` (+ `../../grippers/meshes/pal_pro_gripper/`): the real PAL Pro gripper
  re-rooted as a robosuite gripper (sensors `force_ee` / `torque_ee` on `ft_frame`).

The real `pal_pro_gripper_description` is used (no surrogate). `realsense2_description` is stubbed (head camera frames only).
Licensing: PAL-derived assets are Apache-2.0 (see `pal_tiago_pro/LICENSE`); the rest of the repository is MIT.
"""


def write_docs(pal_root, paths):
    lock = read_lock()
    derived = "\n".join("- [%s](%s) at `%s`" % (c, lock[c][1], lock[c][0]) for c in ov.GIT_CLONES)
    pkg = paths["pkg"]
    with open(os.path.join(pkg, "README.md"), "w") as f:
        f.write(README_PKG.replace("__DERIVED__", derived))
    with open(os.path.join(pkg, "CHANGELOG.md"), "w") as f:
        f.write(CHANGELOG_PKG)
    shutil.copyfile(os.path.join(pal_root, ov.GRIPPER_LICENSE), os.path.join(pkg, "LICENSE"))
    assert hashlib.md5(open(os.path.join(pkg, "LICENSE"), "rb").read()).hexdigest() == LICENSE_MD5
    with open(os.path.join(os.path.dirname(paths["robot_right"]), "README.md"), "w") as f:
        f.write(README_TIAGO)


def out_paths(root):
    return {
        "pkg": os.path.join(root, "robots", "tiago_pro", ov.PACKAGE_DIRNAME),
        "robot_right": os.path.join(root, "robots", "tiago_pro", "robot_right.xml"),
        "gripper": os.path.join(root, "grippers", "pal_pro_gripper.xml"),
        "gripper_meshes": os.path.join(root, "grippers", "meshes", "pal_pro_gripper"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pal-root", default=os.environ.get("PAL_ROOT", "/home/user/pal-robotics"))
    ap.add_argument("--stage1", action="store_true", help="only run stage 1 into --out")
    ap.add_argument("--out", default=None, help="stage-1 output directory (with --stage1)")
    ap.add_argument("--out-root", default=ASSET_ROOT, help="robosuite assets root to generate into (default: the repo)")
    ap.add_argument("--format-with", default=os.path.join(DEFAULT_MENAGERIE, "format_xml.py"))
    ap.add_argument("--check", action="store_true", help="regenerate into a temp dir and byte-diff against the repo files")
    ap.add_argument("--update-lock", action="store_true", help="rewrite sources.lock from the current clones")
    a = ap.parse_args()
    if a.update_lock:
        _require_mujoco()
        sys.path.insert(0, os.path.join(HERE, "shim"))
        open(LOCK_PATH, "w").write(compute_lock(a.pal_root))
    if os.path.exists(LOCK_PATH):
        for c, (sha, _) in read_lock().items():
            assert git_sha(os.path.join(a.pal_root, c)) == sha, "clone %s is at a different commit than sources.lock" % c
    if a.stage1:
        info = stage1(a.pal_root, a.out or os.path.join(tempfile.gettempdir(), "tiago_pro_s1"))
        print({k: v for k, v in info.items() if k not in ("limits", "meshes")})
        return 0
    if a.check:
        tmp = tempfile.mkdtemp(prefix="tiago_pro_check_")
        try:
            cmd = [sys.executable, os.path.abspath(__file__), "--pal-root", a.pal_root, "--out-root", tmp,
                   "--format-with", a.format_with]
            subprocess.check_call(cmd, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            diffs = []
            for key, kind in (("pkg", "dir"), ("robot_right", "file"), ("gripper", "file"), ("gripper_meshes", "dir")):
                new, old = out_paths(tmp)[key], out_paths(a.out_root)[key]
                if kind == "file":
                    if not filecmp.cmp(new, old, shallow=False):
                        diffs.append(old)
                else:
                    stack = [filecmp.dircmp(new, old)]
                    while stack:
                        dc = stack.pop()
                        diffs += [os.path.join(dc.right, f) for f in dc.left_only + dc.right_only + dc.diff_files]
                        stack += list(dc.subdirs.values())
            rd = os.path.join(os.path.dirname(out_paths(tmp)["robot_right"]), "README.md")
            if not filecmp.cmp(rd, os.path.join(os.path.dirname(out_paths(a.out_root)["robot_right"]), "README.md"), shallow=False):
                diffs.append(rd)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print("CHECK:", "OK, regenerated files are byte-identical to the committed ones" if not diffs else "DIFFERENT: %s" % diffs)
        return 1 if diffs else 0
    work = tempfile.mkdtemp(prefix="tiago_pro_s1_")
    try:
        info = stage1(a.pal_root, os.path.join(work, "s1"))
        fmt = load_formatter(a.format_with)
        paths = out_paths(a.out_root)
        stage2(a.pal_root, os.path.join(work, "s1"), info, paths["pkg"], fmt)
        write_docs(a.pal_root, paths)
        rs = Robosuite(paths["pkg"], info)
        rs.robot_right(paths["robot_right"], fmt)
        rs.gripper(paths["gripper"], paths["gripper_meshes"], fmt)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
