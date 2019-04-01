
Installation
------------

Requirements
^^^^^^^^^^^^

xhistogram is compatible with python 3. It requires numpy and, optionally,
xarray.

Installation from Conda Forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install xhistogram along with its dependencies is via conda
forge::

    conda install -c conda-forge xhistogram


Installation from Pip
^^^^^^^^^^^^^^^^^^^^^

An alternative is to use pip::

    pip install xhistogram

This will install the latest release from
`pypi <https://pypi.python.org/pypi>`_.

Installation from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^

xhistogram is under active development. To obtain the latest development version,
you may clone the `source repository <https://github.com/xgcm/xhistogram>`_
and install it::

    git clone https://github.com/xgcm/xhistogram.git
    cd xhistogram
    python setup.py install

or simply::

    pip install git+https://github.com/xgcm/xhistogram.git

Users are encouraged to `fork <https://help.github.com/articles/fork-a-repo/>`_
xhistogram and submit issues_ and `pull requests`_.

.. _dask: http://dask.pydata.org
.. _xarray: http://xarray.pydata.org
.. _issues: https://github.com/xgcm/xhistogram/issues
.. _`pull requests`: https://github.com/xgcm/xhistogram/pulls
