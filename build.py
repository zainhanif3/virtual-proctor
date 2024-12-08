import PyInstaller.__main__
import os
import shutil

def build_app():
    # Clean previous builds
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
        
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # PyInstaller command line arguments
    args = [
        'run.py',  # Your main script
        '--name=VirtualProctor',  # Name of the executable
        '--onedir',  # Create a directory containing the executable
        '--windowed',  # Windows only: do not open a console window
        '--add-data=models;models',  # Include models directory
        '--icon=icon.ico',  # Add an icon (create one first)
        '--noconsole',  # Don't show console window
        '--clean',  # Clean PyInstaller cache
        '--add-binary=models/*;models',  # Include model files
    ]

    # Run PyInstaller
    PyInstaller.__main__.run(args)

if __name__ == "__main__":
    build_app() 