# 1. first, update the version


# 2. run this
rm -r build/
rm -r dist/
rm -r quasinet.egg-info/
rm -r .eggs/
python3 setup.py sdist bdist_wheel


# 3. upload to pypi

# test.pypi
# python3 -m twine upload --repository testpypi dist/*

# for real:
# twine upload dist/*
# -- or --
# python3 -m twine upload dist/*


# download package:
# python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps quasinet==0.0.VERSION