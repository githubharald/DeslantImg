"""
Installs:
    - deslant-img
"""

import codecs

from setuptools import setup

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='deslant-img',
    description='The deslanting algorithm sets text upright in images',
    long_description=README,
    long_description_content_type='text/markdown',
    version='1.0.0',
    url='https://github.com/githubharald/DeslantImg',
    author='Harald Scheidl',
    packages=['deslant_img'],
    install_requires=open('requirements.txt').read().split('\n'),
    entry_points={
        'console_scripts': ['deslant_img=deslant_img.cli:main'],
    }
)
