import os
import sys
import site
from cx_Freeze import setup, Executable

# Increase recursion limit
sys.setrecursionlimit(5000)

# Add 'build' command to arguments if no command is provided
if len(sys.argv) == 1:
    sys.argv.append("build")

# Get site-packages directory
site_packages = site.getsitepackages()[0]

# Dependencies are explicitly listed to ensure all required modules are included
build_exe_options = {
    "packages": [
        "tkinter",
        "cv2",
        "PIL",
        "numpy",
        "ultralytics",
        "torch",
        "yaml",
    ],
    "excludes": [
        "tkinter.test",
        "unittest",
        "email",
        "http",
        "xml",
        "pydoc_data",
    ],
    "include_files": [
        ("models", "models"),
    ],
    "zip_include_packages": ["*"],
    "zip_exclude_packages": [],
    "optimize": 2
}

# Create executable
try:
    base = None
    if sys.platform == "win32":
        base = "Win32GUI"

    setup(
        name="Virtual Proctor",
        version="1.0",
        description="Virtual Proctoring System",
        options={"build_exe": build_exe_options},
        executables=[
            Executable(
                script="run.py",
                base=base,
                target_name="VirtualProctor.exe"
            )
        ]
    )
except Exception as e:
    print(f"An error occurred during setup: {str(e)}")
    sys.exit(1) 