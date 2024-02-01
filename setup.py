from setuptools import setup, find_namespace_packages

setup(
    name='robot_test',
    version='0.1.0',    
    packages=find_namespace_packages(), 
    include_package_data = True,
    #package_dir={'': 'src'},
)
