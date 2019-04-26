# networkx_qnets will run on the .dot files generated in the first step and create a PNG of the qnet
# the default resolution is sufficient for the cchfl example
python3 networkx_qnets.py ./test_output --output_dir ./ --prefixes cchfl
# output also stored there