from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='quantOmics',
    version='1.0.0',
    description='Toolkit for quantitative genetics',
    long_description=long_description,
    url='https://github.com/rbpisupati/quantitativeOmics',
    author=['Rahul Pisupati'],
    author_email='rahul.bharadwaj.p@gmail.com',
    license='GMI',
    classifiers=[
        'Development Status :: 1 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='quantitative genetics',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        "numpy >=1.6.1",
        "scipy >=0.17.0",
        "limix",
        "statsmodels"
    ],
    entry_points={
        'console_scripts': [
            'quantomics=pyquant:main'
        ],
    },
)
