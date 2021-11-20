"""
Installs:
    - DeslantImgPlot
"""

import codecs
import json
from setuptools import setup

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='DeslantImg',
    description='The deslanting algorithm sets text upright in images',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Harald Scheidl',
    url='https://github.com/githubharald/DeslantImg',
    license='MIT',
    py_modules=['main', 'deslant'],
    package_dir={'': 'src/py'},
    install_requires=open('requirements.txt').read().split('\n'),
    entry_points={
        'console_scripts': [
            'DeslantImgPlot=main:main',
        ]
    },
)
