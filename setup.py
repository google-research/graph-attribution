# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for graph_attribution.
This script will install the code as a Python module.
See: https://github.com/google-research/graph-attribution
"""
import pathlib

from setuptools import find_packages, setup

GIT_URL = 'https://github.com/google-research/graph-attribution'

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')
short_description = 'Attribution for GNNs'


install_requires = [
    'tensorflow >= 2.0.0',
    'dm-sonnet >= 2',
    'graph_nets',
    'absl-py',
    'numpy',
    'matplotlib',
    'seaborn',
    'scipy',
    'pandas',
    'tqdm',
    'networkx',
    'ml_collections'
]

setup(
    name='graph_attribution',
    version='1.0.0b',
    description=short_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=GIT_URL,
    author='',
    author_email='bmsanchez@google.com',
    install_requires=install_requires,
    keywords='attribution, gnn, machine, learning, research, xai',
    include_package_data=True,
    packages=find_packages(),
    package_data={},
    project_urls={  # Optional
        'Bug Reports': f'{GIT_URL}/issues',
        'Source': GIT_URL,
    },
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
