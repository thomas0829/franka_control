from setuptools import find_namespace_packages, setup

requirements = [
        "oculus_reader @ git+https://github.com/rail-berkeley/oculus_reader.git"
        ]
setup(
    name='weird_franka',
    version='0.1.0',    
    packages=find_namespace_packages(), 
    include_package_data = True,
    install_requires=requirements
    #package_dir={'': 'src'},
)
