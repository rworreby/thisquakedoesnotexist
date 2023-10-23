#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Robin Worreby",
    author_email='robin.worreby@math.ethz.ch',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="Synthetic Earthquake Data Generation using Generative Adverserial Networks",
    entry_points={
        'console_scripts': [
            'thisquakedoesnotexist=thisquakedoesnotexist.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='thisquakedoesnotexist',
    name='thisquakedoesnotexist',
    packages=find_packages(include=['thisquakedoesnotexist', 'thisquakedoesnotexist.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.ethz.ch/worreby-msc-thesis/thisquakedoesnotexist',
    version='1.0.0',
    zip_safe=False,
)
