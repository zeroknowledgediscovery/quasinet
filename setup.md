```
NAME=quasinet
conda create --name ${NAME} python=3.7
conda activate ${NAME}

conda install jupyter 
pip install --user ipykernel
pip install jupyter_nbextensions_configurator
conda install scikit-learn scipy numpy numba pandas joblib biopython
pip install jupyter_contrib_nbextensions

python -m ipykernel install --user --name=${NAME}
```


# Sphinx

## Installation

```
conda install sphinx
```

## Running Build

```
sphinx-apidoc -f -o docs/source quasinet
make html
```