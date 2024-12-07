from setuptools import setup, find_packages

setup(
    name='vbd',
    version='1.0.0',
    author='Zhiyu Huang and Zixu Zhang ',
    author_email='zixuz@princeton.edu',
    description='Code for Versatile Behavior Diffusion for Generalized Traffic Agent Simulation',
    packages=find_packages(exclude=['examples']),
)
