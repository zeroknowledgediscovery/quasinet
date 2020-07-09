import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Conversion part 1.')

    parser.add_argument(
        '--file1', 
        type=str,
        help='Fasta file that contains the sequence.')
    
    parser.add_argument(
        '--file2', 
        type=str,
        help='Second fasta file that contains the sequence.')
    
    args = parser.parse_args()