import os
import glob

from Bio import SeqIO
import numpy as np

import qnet


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
    for f in glob.glob(basedir):
        f = os.path.basename(f)
        year = int(f.replace('.joblib', '').split('_')[0])
        possible_years.append(year)

    return possible_years

def list_trained_qnets():
    """List the possible qnets we can use.
    """

    raise NotImplementedError

def load_qnet(qnet_type, extra_descriptions):
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
    
    TRAINED_QNET_DIR = 'qnet_trees/'
    qnet_type = qnet_type.lower()
    extra_descriptions  = extra_descriptions.lower()

    all_qnet_types = ['coronavirus', 'influenza']
    
    if qnet_type not in all_qnet_types:
        raise ValueError('`qnet_type`: {} is not in {}'.format(
            qnet_type, all_qnet_types))
                         
    if qnet_type == 'coronavirus':
        corona_extra_descr = ['bat', 'rat', 'game', 'covid19']
        if extra_descriptions not in corona_extra_descr:
            raise ValueError('`extra_descriptions`: {} is not in {}'.format(
                extra_descriptions, corona_extra_descr))

        f = os.path.join(
            TRAINED_QNET_DIR, 
            qnet_type, 
            extra_descriptions + '.joblib')

    elif qnet_type == 'influenza':
        proteins = ['na', 'ha']
        types = ['h1n1', 'h3n2']
        extra_descriptions = extra_descriptions.split(';')
        if len(extra_descriptions) != 3:
            raise ValueError('There must be 3 different descriptions for influenza')

        type_ = extra_descriptions[0]
        if type_ not in types:
            raise ValueError('{} is not in {}'.format(type_, types))

        protein = extra_descriptions[1]
        if protein not in proteins:
            raise ValueError('{} is not in {}'.format(protein, proteins))

        base_dir = os.path.join(
            TRAINED_QNET_DIR, 
            qnet_type, 
            '{}human{}/'.format(type_, protein))

        possible_years = _get_possible_years(base_dir)

        year = extra_descriptions[1]
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

    seq = np.array(str(fastas[0].seq))
    return seq
                         
                         
    