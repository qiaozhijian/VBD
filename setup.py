from setuptools import setup, find_packages

setup(
    name='vbd',
    version='1.0.0',
    author='Zhiyu Huang and Zixu Zhang ',
    author_email='zixuz@princeton.edu',
    description='Code for Versatile Scene-Consistent Traffic Scenario Generation as Optimization with Diffusion',
    packages=find_packages(exclude=['examples']),
)