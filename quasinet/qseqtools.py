import os
import glob

from Bio import SeqIO
import numpy as np

from . import qnet as qnet

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAINED_QNET_DIR = os.path.join(BASE_DIR, 'qnet_trees/')

ALL_QNET_TYPES = ['coronavirus', 'influenza']
CORONA_OPTIONS = ['bat', 'rat', 'game', 'covid19']

INFLUENZA_PROTEINS = ['na', 'ha']
INFLUENZA_TYPES = ['h1n1', 'h3n2']

def _get_possible_years(basedir):
    """Given a base directory for influenza, get the possible years.

    Parameters
    ----------
    basedir : str
        Base directory to search

    Returns
    -------
    possible_years : list
        list of possible years
    """

    possible_years = []
    for f in glob.glob(basedir + "*.joblib"):
        f = os.path.basename(f)
        year = int(f.replace('.joblib', '').split('_')[0])
        possible_years.append(year)

    possible_years.sort()

    return possible_years


def list_trained_qnets():
    """List the possible qnets we can use.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    print('Possible qnets: \n')
    print('`coronavirus` options:')
    print('\t' + '\n\t'.join(CORONA_OPTIONS))
    
    print('\n`influenza` options:')
    influenza_options = []

    for protein in INFLUENZA_PROTEINS:
        for type_ in INFLUENZA_TYPES:
            base_dir = os.path.join(
            TRAINED_QNET_DIR, 
            'influenza', 
            '{}human{}/'.format(type_, protein.upper()))

            possible_years = _get_possible_years(base_dir)
            for year in possible_years:
                influenza_options.append('{};{};{}'.format(type_, protein, year))

    print('\t' + '\n\t'.join(influenza_options))


def load_trained_qnet(qnet_type, extra_descriptions):
    """Load the pre-trained qnet.

    Examples
    ----------
    >>> load_qnet('coronavirus', 'bat')
    >>> load_qnet('influenza', 'h1n1;na;2009')

    Parameters
    ----------
    qnet_type : str
        The type of qnet to load

    extra_descriptions : str
        Extra descriptions for which qnet to load. The descriptions must be
        split by `;` for influenza.

    Returns
    -------
    trained_qnet : qnet.Qnet
        A trained qnet
    """
    
    qnet_type = qnet_type.lower()
    extra_descriptions  = extra_descriptions.lower()
    
    if qnet_type not in ALL_QNET_TYPES:
        raise ValueError('`qnet_type`: {} is not in {}'.format(
            qnet_type, ALL_QNET_TYPES))
                         
    if qnet_type == 'coronavirus':
        if extra_descriptions not in CORONA_OPTIONS:
            raise ValueError('`extra_descriptions`: {} is not in {}'.format(
                extra_descriptions, CORONA_OPTIONS))

        f = os.path.join(
            TRAINED_QNET_DIR, 
            qnet_type, 
            extra_descriptions + '.joblib')

    elif qnet_type == 'influenza':
        extra_descriptions = extra_descriptions.split(';')
        if len(extra_descriptions) != 3:
            raise ValueError('There must be 3 different descriptions for influenza')

        type_ = extra_descriptions[0]
        if type_ not in INFLUENZA_TYPES:
            raise ValueError('{} is not in {}'.format(type_, INFLUENZA_TYPES))

        protein = extra_descriptions[1]
        if protein not in INFLUENZA_PROTEINS:
            raise ValueError('{} is not in {}'.format(protein, INFLUENZA_PROTEINS))

        base_dir = os.path.join(
            TRAINED_QNET_DIR, 
            qnet_type, 
            '{}human{}/'.format(type_, protein.upper()))

        possible_years = _get_possible_years(base_dir)

        year = extra_descriptions[2]

        if not year.isdigit():
            raise ValueError('Year must be an positive integer.')
        
        year = int(year)

        if year not in possible_years:
            raise ValueError('{} is not in {}'.format(year, possible_years))

        f = os.path.join(base_dir, '{}_{}.joblib'.format(year, year + 1))

    else:
        raise ValueError

    trained_qnet = qnet.load_qnet(f)

    return trained_qnet

def load_sequence(file):
    """Load a fasta sequence from file.

    Parameters
    ----------
    file : str
        File of a fasta sequence

    Returns
    -------
    seq : 1d-like array
        A fasta sequence
    """

    fastas = list(SeqIO.parse(file, "fasta"))

    if len(fastas) != 1:
        raise ValueError('Your fasta sequence can only have 1 sequence in it.')

    seq = np.array(list(str(fastas[0].seq)))
    return seq
                         
                         
    