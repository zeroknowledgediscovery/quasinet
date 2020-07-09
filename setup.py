import setuptools
from codecs import open
from os import path
import warnings

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

package_name = 'quasinet'
example_dir = 'examples/'
example_data_dir = example_dir + 'example_data/'

setuptools.setup(
    name=package_name,
    author='zed.uchicago.edu',
    author_email='ishanu@uchicago.edu',
    version = str(version['__version__']),
    packages=setuptools.find_packages(),
    # package_data={'qnet_trees': ['*/*/*'] },
    scripts=[],
    url='https://github.com/zeroknowledgediscovery/',
    license='LICENSE.txt',
    description='Utitilies for constructing and manipulating models for non-local structural dependencies in genomic sequences',
    keywords=[
        'decision trees', 
        'machine learning', 
        'computational biology'],
    download_url='https://github.com/zeroknowledgediscovery/quasinet/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "scikit-learn", 
        "scipy", 
        "numpy", 
        "numba", 
        "pandas",
        "joblib", 
        "biopython"],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6"],
    # data_files=[
    #     ('qnet_example/example1', [
    #         example_dir + "create_qnet.py",
    #         example_dir + "measure_qdistance.py",
    #         example_data_dir + "cchfl_test.csv"
    #         ])
    #     ],
    include_package_data=True)