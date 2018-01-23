#!/usr/bin/env python2
from setuptools import setup
from setuptools import find_packages

setup(
    name="prototype",
    version="0.1",
    description="Robotics Prototype Library",
    author="Chris Choi",
    author_email="chutsu@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.13.0",
        "scipy",
        "jinja2",
        "Pillow",
        "matplotlib",
        "sympy",
        # "pygame",
        "PyOpenGL",
        "opencv-python",
        "autopep8",
        "pyyaml"
    ]
)
