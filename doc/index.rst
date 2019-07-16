xhistogram: Fast, flexible, label-aware histograms for numpy and xarray
=======================================================================

Histograms (a.k.a "binning") are much more than just a visualization tool.
They are the foundation of a wide range of scientific analyses including
[joint] probability distributions and coordinate transformations.
Xhistogram makes it easier to calculate flexible, complex histograms with
multi-dimensional data. It integrates (optionally) with Dask, in order to
scale up to very large datasets and with Xarray, in order to consume and
produce labelled, annotated data structures. It is useful for a wide range of
scientific tasks.


Why a new histogram package?
----------------------------

The main problem with the standard ``histogram`` function in numpy and
dask is that they automatically act over the entire input array (i.e. they
"flatten" the data). Xhistogram allows you to choose which axes / dimensions
you want to preserve and which you want to flatten. It also allows you to
combine N arbitrary inputs to produce N-dimensional histograms.
A good place to start is the :doc:`tutorial`.

Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   tutorial
   api
   contributing
