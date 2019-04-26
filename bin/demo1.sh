# run_qnet_local will iteratively call qNet in a subprocess to generate all of the decision trees and edge files
# those files are stored in test_output
python run_qnet_local.py ./ ./test_output --prefixes cchfl --cutoffs 75 --num_features 400 --min_features 0
# output generated in ../test_output
