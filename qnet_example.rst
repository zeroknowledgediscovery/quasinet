===============
quasinet
===============

.. image:: http://zed.uchicago.edu/logo/logozed1.png
   :height: 800px
   :scale: 50 %
   :alt: alternate text
   :align: center

:Author: ZeD@UChicago <zed.uchicago.edu>
:Description: Infer Non-local Structural Dependencies In Genomic Sequences. Genomic
    sequences are esentially compressed encodings of phenotypic information.
    This package provides a novel set of tools to extract long-range structural
    dependencies in genotypic data that define the phenotypic outcomes.


.. image:: https://zed.uchicago.edu/data/img/q-ltnr.png
   :height: 800px
   :scale: 50 %
   :alt: q-net for long term non-progressor clinical phenotype in HIV-1 infection
   :align: center

**Introduction:**
    This document will walk through an example usage of the **quasinet** python
    package. We will start with an example set of DNA sequences. These sequences
    will consist of 400 basis pairs each. For each of these 400 positions/indices
    we will construct a conditional inference tree. We will then have a decision
    tree for each of these positions where the decisions are informed by the
    basis pairs at other positions. We will then produce some visualizations of the
    Qnetwork.

**Requirements:**
    The quasinet package can be easily installed through pypi.

    .. code-block:: bash

        pip install quasinet

    This should install all required python libraries necessary for this package.
    However, implementations of conditional inference trees are not readily available
    in Python. Thus, this package uses Ctrees implemented in R. We require R
    and two R packages to be installed: **partykit** (1.1-1) and **randomForest** (4.6-14).
    The version of these packages do matter. A third package, **Formula**, is necessary
    for partykit to work. We cannot gurantee the package will work
    with other versions of these R packages. To install R:

    |

    .. code-block:: bash

        apt install r-base-core

    Here are three links to three sources of required R packages.

    1. https://cran.r-project.org/web/packages/Formula/index.html
    2. https://cran.r-project.org/src/contrib/Archive/partykit/
    3. https://cran.r-project.org/web/packages/randomForest/index.html

    All three packages can be installed easily and in the same way. Download the
    zipped source files. Then open R and run:

    .. code-block:: R

        install.packages("/path/to/source.tar.gz", repos=NULL, type="source")
