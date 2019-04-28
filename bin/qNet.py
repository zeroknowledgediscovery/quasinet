#!/usr/bin/python

import numpy as np
import pandas as pd
import subprocess
import os
import sys
from quasinet import mlx
import scipy.stats as stat
import argparse
import warnings
import tempfile
import operator
import multiprocessing as mp
from graphviz import Digraph
import pickle
import glob

warnings.filterwarnings("ignore")
DEBUG=True

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def makeUnique(dict_):
    dict__={}
    keys=list(set(dict_.keys()))
    for key in keys:
        dict__[key]=dict_[key]
    return dict__

#./qNet.py --file /home/ishanu/ZED/Research/mlexpress_/data/qdat11.dat --filex /home/ishanu/ZED/Research/mlexpress_/data/qdat11.dat  --varimp True --response DDR1 --zerodel B --del CELL --importance_threshold 0.24

parser = argparse.ArgumentParser(description='Example with non-optional arguments:\
./dec_tree_2.py --file data.dat --filex data.dat --varimp True --response DDR1 --zerodel B --del CELL --importance_threshold 0.5')

parser.add_argument('--response', dest='RESPONSE', action="store", type=str,nargs='+',
                    default='SPECIES',help="Response Variable")
parser.add_argument('--file', dest='FILE', action="store", type=str,help="train datafile",
                    default='../../data/database_may30_2017/AA_Human_20022003')
parser.add_argument('--filex', dest='FILEx', action="store", type=str,help="test datafile",
                    default='../../data/database_may30_2017/AA_Human_20042005')
parser.add_argument('--ntree', dest='NUMTREE',
                    action="store", type=int,
                    default=300,help="Number of trees in random forest")
parser.add_argument('--cores', dest='CORES',
                    action="store", type=int,
                    default=mp.cpu_count(),help="Number of cores to use in rndom forest")
parser.add_argument('--sample', dest='SAMPLES',
                    action="store", type=int,default=10,help="sample size for columns")
parser.add_argument("--plot", type=str2bool, nargs='?',dest='PLOT_',
                    const=True, default=False,
                    help="Show plot")
parser.add_argument("--varimp", type=str2bool, nargs='?',dest='VARIMP',
                    const=True, default=False,
                    help="Feature importance")
parser.add_argument("--balance", type=str2bool, nargs='?',dest='BALANCE',
                    const=True, default=False,
                    help="Balance class frequency of reposnse variable")
parser.add_argument("--samplefeatures", type=str2bool, nargs='?',dest='SAMPLECOL',
                    const=True, default=False,
                    help="Choose a random sample of features")
parser.add_argument('--del', dest='DELETE',
                    action="store", type=str,nargs='+', default='',help="Deleted features")
parser.add_argument('--inconly', dest='INCLUDEONLY',
                    action="store", type=str,nargs='+',
                    default='',help="Included features, only")
parser.add_argument('--inc', dest='INCLUDE',
                    action="store", type=str,nargs='+', default='',
                    help="Included features")
parser.add_argument("--verbose", type=str2bool, nargs='?',dest='VERBOSE',
                    const=True, default=False,
                    help="Verbose")
parser.add_argument('--treename', dest='TREENAME', action="store", type=str,
                    default='')
parser.add_argument('--zerodel', dest='ZERODEL',
                    action="store", type=str,nargs='+',
                    default='',
                    help="Delete rows where response is in zerodel")
parser.add_argument('--importance_threshold', dest='FEATURE_IMP_THRESHOLD',
                    action="store", type=float,
                    default=0.2,
                    help="Feature importance threshold: default 0.2")
parser.add_argument('--edgefile', dest='EDGEFILE',
                    action="store", type=str,
                    default="edges.txt",
                    help="edges filename")
parser.add_argument('--dotfile', dest='DOTFILE',
                    action="store", type=str,
                    default="edges.dot",
                    help="dot filename")
parser.add_argument('--randomforest', dest='USE_RANDOMFOREST', type=bool,
                    default = False)
parser.add_argument('--tree_prefix', dest='TREE_PREFIX', action='store',
                    type=str, default = '')
parser.add_argument('--output_dir', dest='OUTPUT_DIR', default='.')

results=parser.parse_args()
RESPONSE=results.RESPONSE
FILE=results.FILE
FILEx=results.FILEx
VERBOSE=results.VERBOSE
NUMTREE=results.NUMTREE
CORES=results.CORES
VARIMP=results.VARIMP
PLOT=results.PLOT_
DELETE=results.DELETE
INCLUDE=results.INCLUDE
INCLUDEONLY=results.INCLUDEONLY
TREENAME=results.TREENAME
SAMPLES=results.SAMPLES
BALANCE=results.BALANCE
SAMPLECOL=results.SAMPLECOL
ZERODEL=results.ZERODEL
FEATURE_IMP_THRESHOLD=results.FEATURE_IMP_THRESHOLD
EDGEFILE=results.EDGEFILE
DOTFILE=results.DOTFILE
RESPONSE = map(mlx.nameclean,RESPONSE)
RS = RESPONSE
TREE_PREFIX = results.TREE_PREFIX
OUTPUT_DIR = results.OUTPUT_DIR

if TREE_PREFIX is not '':
    TREE_PREFIX = TREE_PREFIX + '_'

if INCLUDE != "":
    INCLUDE = list(set(list(set(INCLUDE)).extend(RS)))

# if os.path.exists(DOTFILE) and os.path.exists(EDGEFILE):
#     sys.exit()

INPUTFILE_ = ""
edges={}
SOURCES=[]
PROCESSED=[]
DIFF=[]

def getDot(edges,RESPONSE,DOTFILE='out.dot',EDGEFILE=None):
    dot = Digraph()
    for key,values in edges.iteritems():
            if key[0] is not "":
                dot.edge(key[0],key[1])
    # dot.node(RESPONSE[0],shape='circle')
    # dot.node(RESPONSE[0],style='filled')
    # dot.node(RESPONSE[0],fillcolor='red')
    f1=open(DOTFILE,'w+')
    f1.write(dot.source)
    f1.close()

    if EDGEFILE is not None:
        df1=pd.DataFrame.from_dict(edges,orient='index')
        df1.columns=['imp']
        df1=df1[df1.imp>0.0]
        df1.to_csv(EDGEFILE,header=None,sep=",")

    return

def getTree(RS_=[], prefix = []):
    RS_=[RS_]
    edges_={}

    datatrain = mlx.setdataframe(FILE,outname=INPUTFILE_,
                                delete_=DELETE,
                                include_=INCLUDEONLY,
                                select_col=SAMPLECOL,
                                rand_col_sel=SAMPLES,
                                response_var=RS_,
                                balance=BALANCE,
                                zerodel=ZERODEL)

    datatest = mlx.setdataframe(FILEx,
                               include_=INCLUDEONLY,
                               response_var=RS_,
                               zerodel=ZERODEL)

    CT,Pr,ACC,CF,Prx,ACCx,CFx,TR = mlx.Xctree(RESPONSE__=RS_[0],
                                             datatrain__=datatrain,
                                             datatest__=datatest,
                                             VERBOSE=VERBOSE,
                                             TREE_EXPORT=False)

    if TR is not None:
        output = os.path.join(OUTPUT_DIR, '{}{}.dot'.format(TREE_PREFIX, TR.response_var_))
        pkl = os.path.join(OUTPUT_DIR, '{}{}.pkl'.format(TREE_PREFIX, TR.response_var_))

        if not os.path.exists(output):
            mlx.tree_export(TR, outfilename=output, EXEC=False)
        with open(pkl, 'w') as fh:
            pickle.dump(TR, fh)

        sorted_feature_imp\
            = sorted(TR.significant_feature_weight_.items(),
                                    key=operator.itemgetter(1))
        edges_ = {(i[0],RS_[0]):i[1] for i in sorted_feature_imp
                      if i[1] > FEATURE_IMP_THRESHOLD  }
    if not edges_:
        edges_={('',RS_[0]):0.0}

    return edges_


def makeTree(RS_=[], prefix = []):
    '''
    Like get tree but does not bother with edges.
    '''
    RS_=[RS_]
    edges_={}

    datatrain = mlx.setdataframe(FILE,outname=INPUTFILE_,
                                delete_=DELETE,
                                include_=INCLUDEONLY,
                                select_col=SAMPLECOL,
                                rand_col_sel=SAMPLES,
                                response_var=RS_,
                                balance=BALANCE,
                                zerodel=ZERODEL)

    datatest = mlx.setdataframe(FILEx,
                               include_=INCLUDEONLY,
                               response_var=RS_,
                               zerodel=ZERODEL)

    CT,Pr,ACC,CF,Prx,ACCx,CFx,TR = mlx.Xctree(RESPONSE__=RS_[0],
                                             datatrain__=datatrain,
                                             datatest__=datatest,
                                             VERBOSE=VERBOSE,
                                             TREE_EXPORT=False)
    #print "made tree", TR
    if TR is not None:
        output = os.path.join(OUTPUT_DIR, '{}{}.dot'.format(TREE_PREFIX, TR.response_var_))
        pkl = os.path.join(OUTPUT_DIR, '{}{}.pkl'.format(TREE_PREFIX, TR.response_var_))

        if not os.path.exists(output):
            mlx.tree_export(TR, outfilename=output, EXEC=False)
        with open(pkl, 'w') as fh:
            pickle.dump(TR, fh)


def apply_threshold_to_tree(RS_=[], prefix = []):
    edges_={}
    RS_ = [RS_]
    glob_str = OUTPUT_DIR + '/*' + RS_[0] + '.pkl'

    found_files = glob.glob(glob_str)
    if len(found_files) == 0:
        edges_ = {('',RS_[0]):0.0}
        return edges_
    file = found_files[0]
    
    with open(file, 'r') as fh:
        TR = pickle.load(fh)

    sorted_feature_imp\
        = sorted(TR.significant_feature_weight_.items(),
                                key=operator.itemgetter(1))
    edges_ = {(i[0],RS_[0]):i[1] for i in sorted_feature_imp
                  if i[1] > FEATURE_IMP_THRESHOLD  }
    if not edges_:
        edges_={('',RS_[0]):0.0}

    return edges_


def processEdgeUpdate(edges_):
    SOURCES_=[i[0] for  i in edges_.keys() if edges_[i]>0.0]
    PROCESSED_=list(set([i[1] for  i in edges_.keys()]))
    return SOURCES_,PROCESSED_


def apply_threshold(RS,CORES,DOTFILE,EDGEFILE):
    '''
    For use of applying various thresholds to already created trees.
    '''
    pool = mp.Pool(CORES)
    edges = {}
    SOURCES=[]
    PROCESSED=[]
    DIFF=[]
    while RS is not None:

        edges__ = pool.map(apply_threshold_to_tree,RS)

        if DEBUG:
            print edges__

        for edges_ in edges__:
            SOURCES_,PROCESSED_ = processEdgeUpdate(edges_)
            edges.update(edges_)
            PROCESSED.extend(PROCESSED_)
            SOURCES.extend(SOURCES_)
        SOURCES=list(set(SOURCES))
        PROCESSED=list(set(PROCESSED))
        DIFF = diff(SOURCES,PROCESSED)
        DIFF = diff(DIFF,PROCESSED)
        edges = makeUnique(edges)

        if len(DIFF)>0:
            RS=DIFF
        else:
            RS=None

        if DEBUG:
            print "CURRENT RS--> ", RS

        getDot(edges,RESPONSE,
               DOTFILE=DOTFILE,EDGEFILE=EDGEFILE)

    pool.close()
    pool.join()

def printf(RS):
    print RS

#########Make Trees only ###############
'''
pool = mp.Pool(CORES)

pool.map(printf,RS)
pool.close()
pool.join()
'''

#########Apply threshold only#########
#apply_threshold(RS,CORES,DOTFILE,EDGEFILE)



#########Make Trees and apply threshold.#################

pool = mp.Pool(CORES)
while RS is not None:

    edges__ = pool.map(getTree,RS)

    if DEBUG:
        print edges__

    for edges_ in edges__:
        SOURCES_,PROCESSED_ = processEdgeUpdate(edges_)
        edges.update(edges_)
        PROCESSED.extend(PROCESSED_)
        SOURCES.extend(SOURCES_)
    SOURCES=list(set(SOURCES))
    PROCESSED=list(set(PROCESSED))
    DIFF = diff(SOURCES,PROCESSED)
    DIFF = diff(DIFF,PROCESSED)
    edges = makeUnique(edges)

    if len(DIFF)>0:
        RS=DIFF
    else:
        RS=None

    if DEBUG:
        print "CURRENT RS--> ", RS

    getDot(edges,RESPONSE,
           DOTFILE=DOTFILE,EDGEFILE=EDGEFILE)

pool.close()
pool.join()
