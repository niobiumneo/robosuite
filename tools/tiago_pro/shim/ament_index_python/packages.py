"""Minimal stand-in for ament_index_python.packages (used by xacro 2.x for $(find pkg)).

Package -> directory map comes from the env var TIAGO_PKG_PATHS (path1:path2:..., each path is a
package dir whose basename is the package name) -- no ROS install needed.
"""
import os


class PackageNotFoundError(KeyError):
    pass


def _map():
    m = {}
    for p in os.environ.get("TIAGO_PKG_PATHS", "").split(os.pathsep):
        if p:
            m[os.path.basename(p.rstrip("/"))] = p
    return m


def get_package_share_directory(package_name, print_warning=True):
    m = _map()
    if package_name not in m:
        raise PackageNotFoundError("package '%s' not found (TIAGO_PKG_PATHS=%s)"
                                   % (package_name, sorted(m)))
    return m[package_name]
