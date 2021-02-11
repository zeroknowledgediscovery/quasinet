Contributing
====================

To build documentation::
    sphinx-apidoc -f -o docs/source quasinet 
    cd docs 
    make html 
    git add . -f

To run tests::
    python -m unittest tests.test_quasinet 


