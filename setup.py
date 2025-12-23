from setuptools import find_namespace_packages, setup
import os

requirements = [
    "oculus_reader @ git+https://github.com/rail-berkeley/oculus_reader.git"
]

# Optionally add openpi-client from local path if it exists
openpi_client_path = os.path.expanduser("~/thomas/openpi/packages/openpi-client")
if os.path.exists(openpi_client_path):
    requirements.append(f"openpi-client @ file://{openpi_client_path}")

setup(
    name="weird_franka",
    version="0.1.0",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=requirements,
    # package_dir={'': 'src'},
)
