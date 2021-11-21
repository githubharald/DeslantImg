"""
Installs:
    - deslant-img
"""

import codecs

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

from setuptools import setup

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
    scripts=['scripts/deslant_img.py'],
)
