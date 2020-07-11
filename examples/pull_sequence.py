"""This is a script that pulls a sequence from NCBI, and downloads it

example usage:
    python pull_sequence.py \
        --organism "coronavirus" \
        --accession "LC528233" \
        --outfile "LC528233.fasta"

    python pull_sequence.py \
        --organism "influenza A HA" \
        --accession "KP456738" \
        --outfile "KP456738.fasta"
"""

import pickle
import argparse

from Bio import Entrez
from Bio import SeqIO
import numpy as np
import pandas as pd

def load_pickled(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def save_pickled(item, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(item, f)

def getRecord(search_term, retmax=10, database="nucleotide"):
    """Get records from NCBI based on the search term.
    """
    
    Entrez.email = ""
    handle = Entrez.esearch(
        db=database, 
        term=[search_term], 
        retmax=retmax)

    record = Entrez.read(handle)
    handle.close()
    handle = Entrez.efetch(db=database, id=record["IdList"], retmode="xml")
    records = Entrez.read(handle)
    
    return records

def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
                break
            else:
                for child_val in item_generator(v, lookup_key):
                    yield child_val
                    break
    elif isinstance(json_input, list):
        for item in json_input:
            for item_val in item_generator(item, lookup_key):
                yield item_val
                break

def procSequence(
    records,
    begIndex,
    endIndex,
    type_='nucleotide',
    N=10000,
    LMAX=35000):  


    S=[]
    ACC=[]
    count=0
    for i in records:
        beg=[ x for x in item_generator(i,'GBInterval_from')][0]
        end=[ x for x in item_generator(i,'GBInterval_to')][0]

        if type_ == 'nucleotide':
            seq=[ x for x in item_generator(i,'GBSeq_sequence')][0]
        elif type_ == 'protein':
            seq = []
            for j in i['GBSeq_feature-table']:
                if 'GBFeature_quals' in j:
                    for k in j['GBFeature_quals']:
                        if k['GBQualifier_name'] == 'translation':
                            seq.append(k['GBQualifier_value'])
            seq = seq[0]
        else:
            raise ValueError('Not an available type: {}'.format(type_))

        acc=[ x for x in item_generator(i,'GBSeq_primary-accession')][0]
        
        xbeg=''.join('x' for i in np.arange(int(beg)))
        xend=''.join('x' for i in np.arange(LMAX-int(end)))
        seq=xbeg+seq+xend
        seq=seq[begIndex:endIndex]
        S=np.append(S,seq)
        ACC=np.append(ACC,acc)
        if count > N:
            break
        else:
            count=count+1
            
    SF=pd.DataFrame([list(x) for x in S]).replace('x',np.nan)
    SF['accession']=ACC
    SF=SF.dropna(how='all',axis=1)
    
    return SF


def pull_sequence(organism, accession):
    """Pull the sequence from NCBI given the organism and the accession name.
    """

    coronavirus_orgs = ['coronavirus']
    influenza_orgs = ['influenza A NA', 'influenza A HA']
    all_organisms = coronavirus_orgs + influenza_orgs

    if organism not in all_organisms:
        raise ValueError('Your organism must be one of {}'.format(all_organisms))

    if organism in coronavirus_orgs:
        record = getRecord(accession)
    elif organism in influenza_orgs:
        record = getRecord(accession)
    else:
        raise ValueError

    if len(record) == 0:
        raise ValueError('We could not find a record in NCBI matching: {}'.format(accession))

    if organism == 'coronavirus':
        df = procSequence(
            record,
            begIndex=21563,
            endIndex=25384)
    elif organism == 'influenza A HA':
        df = procSequence(
            record,
            begIndex=0,
            endIndex=550,
            type_='protein')

    elif organism == 'influenza A NA':
        df = procSequence(
            record,
            begIndex=0,
            endIndex=450,
            type_='protein')
    else:
        raise ValueError

    if df.shape[0] == 0:
        raise ValueError('There is no data for this accession: {}'.format(accession))
    elif df.shape[0] > 1:
        raise ValueError('There are multiple sequences corresponding to this accession: {}'.format(accession))

    seq = ''.join(df.drop(['accession'], axis=1).iloc[0].values)


    return seq

def dump_sequence(seq, accession, file):
    with open(file, 'w+') as f:
        f.write('>' + accession)
        f.write('\n' + seq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Pull sequence from NCBI.')

    parser.add_argument(
        '--organism', 
        type=str,
        help='Organism to pull.')

    parser.add_argument(
        '--accession', 
        type=str,
        help='Accession name.')
    
    parser.add_argument(
        '--outfile', 
        type=str,
        help='File to save pulled sequence.')

    args = parser.parse_args()

    seq = pull_sequence(args.organism, args.accession)
    
    dump_sequence(seq, args.accession, args.outfile)