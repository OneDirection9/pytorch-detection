from __future__ import absolute_import, division, print_function

import os.path as osp
from setuptools import find_packages, setup


def get_version():
    init_py_path = osp.join(osp.abspath(osp.dirname(__file__)), 'det', '__init__.py')
    with open(init_py_path, 'r') as f:
        version_line = [l.strip() for l in f.readlines() if l.startswith('__version__')][0]
    version = version_line.split('=')[-1].strip().strip('"\'')
    return version


def get_readme():
    with open('README.md', 'r') as f:
        content = f.read()
    return content


install_requires = [
    'numpy',
    'pyyaml',
    'six',
    'opencv-python==4.1.0.25',
    'torch>=1.4.0',
    'torchvision>=0.5.0',
]

extras_require = {}

setup(
    name='pytorch-detection',
    version=get_version(),
    description='Platform for object detection and segmentation',
    long_description=get_readme(),
    keywords='computer vision',
    packages=find_packages(include=('det.*',)),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
    author='Zhipeng Han',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    extras_require=extras_require,
)
