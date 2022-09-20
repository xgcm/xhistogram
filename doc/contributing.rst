Contributor Guide
=================

This package is in very early stages. Lots of work is needed.

You can help out just by using ``xhistogram`` and reporting
`issues <https://github.com/xgcm/xhistogram/issues>`__.

The following sections cover some general guidelines for maintainers and
contributors wanting to help develop ``xhistogram``.


Feature requests, suggestions and bug reports
---------------------------------------------

We are eager to hear about any bugs you have found, new features you
would like to see and any other suggestions you may have. Please feel
free to submit these as `issues <https://github.com/xgcm/xhistogram/issues>`__.

When suggesting features, please make sure to explain in detail how
the proposed feature should work and to keep the scope as narrow as
possible. This makes features easier to implement in small PRs.

When report bugs, please include:

* Any details about your local setup that might be helpful in
  troubleshooting, specifically the Python interpreter version, installed
  libraries, and ``xhistogram`` version.
* Detailed steps to reproduce the bug, ideally a `Minimal, Complete and
  Verifiable Example <http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`__.
* If possible, a demonstration test that currently fails but should pass
  when the bug is fixed.


Write documentation
-------------------
Adding documentation is always helpful. This may include:

* More complementary documentation. Have you perhaps found something unclear?
* Docstrings.
* Example notebooks of ``xhistogram`` being used in real analyses.

The ``xhistogram`` documentation is written in reStructuredText. You
can follow the conventions in already written documents. Some helpful guides
can be found
`here <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`__ and
`here <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`__.

When writing and editing documentation, it can be useful to see the resulting
build without having to push to Github. You can build the documentation locally
by running::

    $ # Install the packages required to build the docs in a conda environment
    $ conda env create -f doc/environment.yml
    $ conda activate xhistogram_doc_env
    $ # Install the latest xhistogram
    $ pip install --no-deps -e .
    $ cd doc/
    $ make html

This will build the documentation locally in ``doc/_build/``. You can then open
``_build/html/index.html`` in your web browser to view the documentation. For
example, if you have ``xdg-open`` installed::

    $ xdg-open _build/html/index.html

To lint the reStructuredText documentation files run::

    $ doc8 doc/*.rst


Preparing Pull Requests
-----------------------
#. Fork the
   `xhistogram GitHub repository <https://github.com/xgcm/xhistogram>`__.  It's
   fine to use ``xhistogram`` as your fork repository name because it will live
   under your username.

#. Clone your fork locally, connect your repository to the upstream (main
   project), and create a branch to work on::

    $ git clone git@github.com:YOUR_GITHUB_USERNAME/xhistogram.git
    $ cd xhistogram
    $ git remote add upstream git@github.com:xgcm/xhistogram.git
    $ git checkout -b your-bugfix-feature-branch-name master

   If you need some help with Git, follow
   `this quick start guide <https://git.wiki.kernel.org/index.php/QuickStart>`__

#. Install dependencies into a new conda environment::

    $ conda env create -f ci/environment-3.9.yml
    $ conda activate xhistogram_test_env

#. Install xhistogram using the editable flag (meaning any changes you make to
   the package will be reflected directly in your environment)::

    $ pip install --no-deps -e .

#. Start making your edits. Please try to type annotate your additions as
   much as possible. Adding type annotations to existing unannotated code is
   also very welcome. You can read about Python typing
   `here <https://mypy.readthedocs.io/en/stable/getting_started.html#function-signatures-and-dynamic-vs-static-typing>`__.

#. Break your edits up into reasonably sized commits::

    $ git commit -a -m "<commit message>"
    $ git push -u

   It can be useful to manually run `pre-commit <https://pre-commit.com>`_ as you
   make your edits. ``pre-commit`` will run checks on the format and typing of
   your code and will show you where you need to make changes. This will mean
   your code is more likely to pass the CI checks when you push it::

    $ pip install pre_commit # you only need to do this once
    $ pre-commit run --all-files

#. Run the tests (including those you add to test your edits!)::

    $ pytest xhistogram

   You can also test that your contribution and tests increased the test coverage::

    $ coverage run --source xhistogram -m py.test
    $ coverage report

#. Add a new entry describing your contribution to the :ref:`release-history`
   in ``doc/contributing.rst``. Please try to follow the format of the existing
   entries.

#. Submit a pull request through the GitHub `website <https://github.com/xgcm/xhistogram>`__.

   Note that you can create the Pull Request while you're working on your PR.
   The PR will update as you add more commits. ``xhistogram`` developers and
   contributors can then review your code and offer suggestions.


.. _release-history:

Release History
---------------

v0.3.2
~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix bug producing TypeError when `weights` is provided with
  `keep_coords=True` :issue:`78`. By
  `Dougie Squire <https://github.com/dougiesquire>`_.
- Raise TypeError when `weights` is a dask array and bin edges are
  not explicitly provided :issue:`12`. By
  `Dougie Squire <https://github.com/dougiesquire>`_.

v0.3.1
~~~~~~~~~~~~~~~~~~~~~~~~~

- Add DOI badge and CITATION.cff. By
  `Julius Busecke <https://github.com/jbusecke>`_.

v0.3.0
~~~~~~~~~~~~~~~~~~~~~~~~~

- Add support for histograms over non-float dtypes (e.g. datetime
  objects) :issue:`25`. By
  `Dougie Squire <https://github.com/dougiesquire>`_.
- Refactor histogram calculation to use dask.array.blockwise
  when input arguments are dask arrays, resulting in significant
  performance improvements :issue:`49`. By
  `Ryan Abernathy <https://github.com/rabernat>`_,
  `Tom Nicholas <https://github.com/TomNicholas>`_ and
  `Gabe Joseph <https://github.com/gjoseph92>`_.
- Fixed bug with density calculation when NaNs are present :issue:`51`.
  By `Dougie Squire <https://github.com/dougiesquire>`_.
- Implemented various options for users for providing bins to
  xhistogram that mimic the numpy histogram API. This included
  adding a range argument to the xhistogram API :issue:`13`.
  By `Dougie Squire <https://github.com/dougiesquire>`_.
- Added a function to check if the object passed to xhistogram is an
  xarray.DataArray and if not, throw an error. :issue:`14`.
  By `Yang Yunyi <https://github.com/Badboy-16>`_.

v0.2.0
~~~~~~

- Added FutureWarning for upcoming changes to core API :issue:`13`.
  By `Dougie Squire <https://github.com/dougiesquire>`_.
- Added documentation on how to deal with NaNs in weights :issue:`26`.
  By `Shanice Bailey <https://github.com/stb2145>`_.
- Move CI to GitHub actions :issue:`32`.
  By `James Bourbeau <https://github.com/jrbourbeau>`_.
- Add documentation for contributors.
  By `Dougie Squire <https://github.com/dougiesquire>`_.
- Add type checking with mypy :issue:`32`.
  By `Dougie Squire <https://github.com/dougiesquire>`_.

v0.1.3
~~~~~~

- Update dependencies to exclude incompatible dask version :issue:`27`.
  By `Ryan Abernathey <https://github.com/rabernat>`_.

v0.1.2
~~~~~~

- Aligned definition of ``bins`` with ``numpy.histogram`` :issue:`18`.
  By `Dougie Squire <https://github.com/dougiesquire>`_.

v0.1.1
~~~~~~

Minor bugfix release

- Imroved documentation examples.
  By `Dhruv Balwada <https://github.com/dhruvbalwada>`_.
- Fixed issue :issue:`5` related to incorrect dimension order
  and dropping of dimension coordinates.
  By `Ryan Abernathey <https://github.com/rabernat>`_.

v0.1
~~~~

First release
