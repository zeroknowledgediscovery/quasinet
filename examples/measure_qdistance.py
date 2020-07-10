"""This is an example how to use measure qdistances using pretrained qnets.
"""

from quasinet import qseqtools as qt
from quasinet import qnet

DATA_DIR = 'example_data/'

# to list all the available trained qnets
qt.list_trained_qnets()

# load the sequences from fasta files
seq1 = qt.load_sequence(DATA_DIR + 'influenza1.fasta')
seq2 = qt.load_sequence(DATA_DIR + 'influenza2.fasta')

# load qnet
influenza_qnet = qt.load_trained_qnet('influenza', 'h1n1;ha;2009')

# compute qdistance
qdist = qnet.qdistance(seq1, seq2, influenza_qnet, influenza_qnet) 

# compute membership degree
qnet_membership = qnet.membership_degree(seq1, influenza_qnet)
