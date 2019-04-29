===============
quasinet
===============

.. figure:: https://img.shields.io/pypi/dm/quasinet.svg
   :alt: quasinet PyPI Downloads

.. figure:: https://badge.fury.io/py/quasinet.svg
   :alt: ehrzero PyPI Downloads

.. image:: http://zed.uchicago.edu/logo/logozed1.png
   :height: 400px
   :scale: 50 %
   :alt: alternate text
   :align: center


.. class:: no-web no-pdf

:Author: ZeD@UChicago <zed.uchicago.edu>
:Description: Infer Non-local Structural Dependencies In Genomic Sequences. Genomic sequences are esentially compressed encodings of phenotypic information. This package provides a novel set of tools to extract long-range structural dependencies in genotypic data that define the phenotypic outcomes. The key capabilities implemented here are as follows: 1. computing the q-net given a databse of nucleic acid sequences, which is a family of conditional inference trees capturing the predictability of each nucleotide position given the rest of the genome. 2. Computing a structure-aware evolution-adaptive notion of distance between genomes, which demonstrably is much more biologically relevant compared to the standard edit distance 3. Ability to draw samples in-silico, that have a high probability of being biologically correct. For example, given a database of HIV sequences, we can generate a new genomic sequence, which has a high probability of being a valid encoding of a HIV virion. The constructed q-net for long term non-progressor clinical phenotype in HIV-1 infection is shown below.


.. image:: https://zed.uchicago.edu/data/img/q-ltnr.png
   :height: 400px
   :scale: 50 %
   :alt: q-net for long term non-progressor clinical phenotype in HIV-1 infection
   :align: center
