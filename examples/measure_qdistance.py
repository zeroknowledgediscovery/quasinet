"""This is an example how to use the pretrained qnets for distance and membership
degree calculation.
"""

from quasinet import qseqtools as qt
from quasinet import qnet

DATA_DIR = 'example_data/'

# to list all the available trained qnets
qt.list_trained_qnets()


# ------------------- #
# Influenza example
# ------------------- #

# load the sequences from fasta files
seq1 = qt.load_sequence(DATA_DIR + 'influenza1.fasta')
seq2 = qt.load_sequence(DATA_DIR + 'influenza2.fasta')

# load influenza h1n1 HA 2009 qnet
influenza_qnet = qt.load_trained_qnet('influenza', 'h1n1;ha;2009')

# compute qdistance between sequence 1 and sequence 2
qdist = qnet.qdistance(seq1, seq2, influenza_qnet, influenza_qnet) 

# compute membership degree of sequence 1 with respect to the qnet
qnet_membership = qnet.membership_degree(seq1, influenza_qnet)


# ------------------- #
# Coronavirus example
# ------------------- #

seq1 = qt.load_sequence(DATA_DIR + 'covid19_1.fasta')
seq2 = qt.load_sequence(DATA_DIR + 'covid19_2.fasta')

coronavirus_qnet = qt.load_trained_qnet('coronavirus', 'covid19')

qdist = qnet.qdistance(seq1, seq2, coronavirus_qnet, coronavirus_qnet) 

qnet_membership = qnet.membership_degree(seq1, coronavirus_qnet)