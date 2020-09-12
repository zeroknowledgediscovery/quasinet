Contributing
====================

To build documentation::
    sphinx-apidoc -f -o docs/source quasinet 
    cd docs 
    make html 

To run tests::
    python -m unittest tests.test_quasinet 


