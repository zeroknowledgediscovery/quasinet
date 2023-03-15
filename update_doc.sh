#!/bin/bash

sphinx-apidoc -f -o docs/source quasinet
cd docs
make html
git add . -f
