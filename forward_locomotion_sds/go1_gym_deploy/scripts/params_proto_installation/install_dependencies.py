import os
import subprocess

def install_local_dependencies(wheel_dir):
    """
    Install all local wheel files from a specified directory.
    """
    # List of all dependencies to install from local .whl files
    required_packages = [
        'typing',
        'termcolor',     # Required by params_proto
        'waterbear',     # Required by params_proto
        'expandvars',    # Required by params_proto
        'argcomplete',    # Ensure compatible version for Python 3.6
        'params_proto'  # Main package
    ]

    # Iterate over each required package
    for package in required_packages:
        # Install from local .whl files
        for filename in os.listdir(wheel_dir):
            if filename.startswith(package) and filename.endswith('.whl') :
                wheel_path = os.path.join(wheel_dir, filename)
                print(f"Installing {filename}...")
                subprocess.check_call(['pip3', 'install', '--no-index', '--find-links=' + wheel_dir, wheel_path])
            if filename.startswith(package) and filename.endswith('.gz') :
                wheel_path = os.path.join(wheel_dir, filename)
                print(f"Installing {filename}...")
                subprocess.check_call(['pip3', 'install', '--no-index', '--find-links=' + wheel_dir, wheel_path])
                


if __name__ == "__main__":
    # Define the directory where the .whl files are located
    wheel_directory = '.'  # Replace with actual path

    # Install all the dependencies from the local .whl files
    install_local_dependencies(wheel_directory)
