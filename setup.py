from setuptools import find_packages
from distutils.core import setup

setup(
    name='masters',
    version='1.0.0',
    author='Jeffrey Li',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='zcabbli@ucl.ac.uk',
    install_requires=[
        "openai==0.28.0",
        "numpy<1.24",
        "av",
        'matplotlib',
        "torch==2.4.0",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "stable_baselines3",
        "hydra-core",
        "gpustat",
        "wandb",
        "opencv-python",
        "cloudpickle==1.3.0"
    ]
)
