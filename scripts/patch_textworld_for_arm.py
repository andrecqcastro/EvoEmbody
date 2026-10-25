#!/usr/bin/env python3
"""
Builds a textworld wheel with its Inform7 install step removed and installs it
into the given Python environment.

    scripts/patch_textworld_for_arm.py /path/to/env/bin/python

textworld's setup.py downloads an Inform7 CLI archive at install time, and that
archive has no aarch64 Linux binaries (microsoft/TextWorld#310). Inform7 is
only used by textworld's game generator; ALFWorld ships pre-compiled games that
are played through Jericho, so skipping it is safe here.

The built wheel is cached per target Python version, because textworld builds a
platform-specific wheel tagged with the interpreter that built it.
"""
import glob
import os
import subprocess
import sys
import tarfile

CACHE_ROOT = os.environ.get("TEXTWORLD_PATCH_CACHE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "textworld_patch")

OLD_PRE_INSTALL = (
    "def _pre_install(dir):\n"
    "    from subprocess import check_call\n"
    "    check_call(['./setup.sh'], shell=True, cwd=os.getcwd())\n"
)
NEW_PRE_INSTALL = (
    "def _pre_install(dir):\n"
    "    # Inform7 install skipped by scripts/patch_textworld_for_arm.py.\n"
    "    pass\n"
)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: patch_textworld_for_arm.py /path/to/env/bin/python")
    # Absolute, because the wheel build below runs from the extracted source directory.
    target_python = os.path.abspath(sys.argv[1])

    # The source download is shared; the built wheel is cached per Python version.
    python_tag = subprocess.check_output(
        [target_python, "-c", "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"],
        text=True).strip()

    os.makedirs(CACHE_ROOT, exist_ok=True)
    src_dir = os.path.join(CACHE_ROOT, "src")
    wheel_dir = os.path.join(CACHE_ROOT, python_tag, "wheel")

    existing_wheels = glob.glob(os.path.join(wheel_dir, "textworld-*.whl"))
    if not existing_wheels:
        os.makedirs(src_dir, exist_ok=True)
        print("Downloading textworld's source distribution (matching alfworld's own "
              "textworld[pddl]>=1.6.1 requirement)...")
        subprocess.check_call([
            target_python, "-m", "pip", "download", "textworld[pddl]",
            "--no-deps", "--no-binary", "textworld", "-d", src_dir,
        ])
        tarball = glob.glob(os.path.join(src_dir, "textworld-*.tar.gz"))[0]
        with tarfile.open(tarball) as tf:
            tf.extractall(src_dir)
        tw_dir = [d for d in glob.glob(os.path.join(src_dir, "textworld-*")) if os.path.isdir(d)][0]

        setup_py = os.path.join(tw_dir, "setup.py")
        original = open(setup_py, encoding="utf-8").read()
        if OLD_PRE_INSTALL not in original:
            sys.exit(f"FAIL: textworld's setup.py at {setup_py} no longer matches the expected "
                     f"_pre_install pattern this patch targets; it may have changed upstream. "
                     f"Update OLD_PRE_INSTALL in this script to match before re-running.")
        open(setup_py, "w", encoding="utf-8").write(original.replace(OLD_PRE_INSTALL, NEW_PRE_INSTALL))
        print(f"Patched {setup_py}: Inform7 CLI install step removed.")

        os.makedirs(wheel_dir, exist_ok=True)
        print("Building the patched wheel...")
        subprocess.check_call([target_python, "-m", "pip", "wheel", ".", "--no-deps", "-w", wheel_dir],
                              cwd=tw_dir)
        existing_wheels = glob.glob(os.path.join(wheel_dir, "textworld-*.whl"))
        if not existing_wheels:
            sys.exit("FAIL: wheel build did not produce a textworld-*.whl.")

    print(f"Installing patched wheel: {existing_wheels[0]}")
    subprocess.check_call([target_python, "-m", "pip", "install", existing_wheels[0]])


if __name__ == "__main__":
    main()
