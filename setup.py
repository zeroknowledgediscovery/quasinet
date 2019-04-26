from setuptools import setup
from codecs import open
from os import path
import warnings

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


warnings.warn("WARNING: This package uses rpy2 as well as R. Please install R (3.3+). \
    The R packages: partykit(1.1-1) and randomForest(4.6-14) is also necessary.")



setup(
    name='quasinet',
    author='zed.uchicago.edu',
    author_email='ishanu@uchicago.edu',
    version = str(version['__version__']),
    packages=['mlexpress'],
    scripts=[],
    url='https://github.com/zeroknowledgediscovery/',
    license='LICENSE.txt',
    description='Decision tree',
    keywords=['decision','trees'],
    download_url='https://github.com/zeroknowledgediscovery/Cynet/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=["numpy >= 1.6","pandas >= 0.22.0","matplotlib >= 2.0.2","rpy2 == 2.8.6",\
    "scipy>=1.1.0","ascii_graph >= 1.5.1", 'graphviz>=0.10.1'],
    python_requires='==2.7.*',
    classifiers=[\
    'Development Status :: 4 - Beta',
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 2.7"],
    data_files=[('qnet_example/',['bin/cchfl_test.csv','bin/cchfl_train.csv','bin/demo1.sh','bin/demo2.sh',\
    							  'bin/networkx_qnets.py','bin/qNet.py','bin/README.md','bin/run_qnet_local.py'])],
    include_package_data=True)
