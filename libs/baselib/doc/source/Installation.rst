Installation
============

```
pip install sphinx
pip install sphinx_rtd_theme
```

Initialization
--------------

```
sphinx-quickstart
```

Generate auto-documentation
---------------------------

```
cd project_root
sphinx-apidoc -o doc/source/ baselib/
```

Build
-----

```
make html
```

Creating docstrings
-------------------

https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
