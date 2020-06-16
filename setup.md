```
conda create --name quasinet python=3.7
conda activate quasinet

conda install jupyter 
pip install --user ipykernel
pip install jupyter_nbextensions_configurator
conda install scikit-learn scipy numpy numba pandas joblib
pip install jupyter_contrib_nbextensions

python -m ipykernel install --user --name=quasinet
```