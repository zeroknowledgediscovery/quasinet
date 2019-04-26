import subprocess
import multiprocessing as mp
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to folder containing train and test files (named prefix_train.csv and prefix_test.csv)')
parser.add_argument('output_directory', help='Path for where to store output files (.dot, .pkl, .dat)')
parser.add_argument('--prefixes', nargs='+', help='Subtype prefixes (e.g. SPhiv, Phiv, cchfl)', default=['cchfl'])
parser.add_argument('--cutoffs', nargs='+', type=float, help='Confidence thresholds (0-100)', default=[75])
parser.add_argument('--num_features', nargs='+', type=int, help='Number of features per sequence specified in prefixes')
parser.add_argument('--min_features', nargs='+', type=int, help='Features to skip')
results = parser.parse_args()

results.cutoffs = [x/100 for x in results.cutoffs]

data_path = os.path.abspath(results.data_path)
output_directory = os.path.abspath(results.output_directory)
prefixes = results.prefixes
cutoffs = results.cutoffs
num_features = results.num_features
min_features = results.min_features[0]

assert(len(prefixes) == len(cutoffs) == len(num_features)), \
"When specifying prefixes, cutoffs, or features, you must specify all three with their corresponding values"

if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

command = "python " + os.path.abspath('qNet.py') + " --file {} --filex {} --varimp True --response {} --importance_threshold {} --edgefile {}.dat --dotfile {}.dot --tree_prefix {} --output_dir {}"


num_cores = mp.cpu_count()

for i, prefix in enumerate(prefixes):
    features = list(range(num_features[i]))
    feature_groups = [features[i:i + num_cores] for i in range(min_features, len(features), num_cores)]
    train_name = '{}_train.csv'.format(prefix)
    test_name = '{}_test.csv'.format(prefix)

    train_file = os.path.abspath(os.path.join(data_path, train_name))
    test_file = os.path.abspath(os.path.join(data_path, test_name))
    for cutoff in cutoffs:
        for group in feature_groups:
            current_command = command.format(
                train_file,
                test_file,
                ' '.join([str(x) for x in group]),
                cutoff,
                os.path.join(output_directory, '{}_{}-{}_{}'.format(prefix, group[0], group[-1], int(cutoff * 100))),
                os.path.join(output_directory, '{}_{}-{}_{}'.format(prefix, group[0], group[-1], int(cutoff * 100))),
                '{}_{}'.format(int(cutoff * 100), prefix),
                output_directory,
            )
            print(current_command)
            subprocess.call([current_command], shell = True)

