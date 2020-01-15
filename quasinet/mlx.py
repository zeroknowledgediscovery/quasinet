import string
import random
import subprocess
import os
import sys
import decimal
import operator
import glob
import pickle
import copy
import math
import multiprocessing

import numpy as np
import pandas as pd
import rpy2.rinterface as rinterface
rinterface.set_initoptions(
    ("rpy2", "--max-ppsize=500000", '--no-save', '--no-restore', '--quiet'))
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
import scipy.stats as stat

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()

#------------------------------------------
from ascii_graph import Pyasciigraph
from ascii_graph.colors import *
from ascii_graph.colordata import vcolor
from ascii_graph.colordata import hcolor
#------------------------------------------
import pprint
pp = pprint.PrettyPrinter(indent=4)

DEBUG=False
DEBUG__=False
DEBUG___=False

WHITE   = "\033[0;37m"
RED   = "\033[0;31m"
YELLOW  = "\033[0;33"
PURPLE  = "\033[0;35m"
BLUE  = "\033[0;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"

WHITE   = ""
RED   = ""
YELLOW  = ""
PURPLE  = ""
BLUE  = ""
CYAN  = ""
GREEN = ""
RESET = ""
BOLD    = ""
REVERSE = ""

#------------------------------------------
filled=True #///added
rounded=True
leaves_parallel=True
rotate=False
ranks = {'leaves': []}
        # The colors to render each node with
colors = {'bounds': None}
out_file = open('out.dot', "w")

#------------------------------------------

isnan = lambda x: type(x) is float and math.isnan(x)

#------------------------------------------

class tree_(object):

    def __init__(self, nodes=[],feature={},leaf_={},
                 children={},
                 children_left={},children_right={},
                 edge_cond_={},error_node_={},pvalue_node={},
                 CLASSES_={},numpass={},terminal_prob={},
                 class_pred={},decision_rules={},sig_f_wt={},
                 ACC_=None,ACCx_=None,resp_=None):
        self.nodes=nodes
        self.feature=feature
        self.TREE_LEAF=leaf_
        self.children=children
        self.children_right=children_right
        self.children_left=children_left
        self.edge_cond_=edge_cond_
        self.error=error_node_
        self.pvalue=pvalue_node
        self.CLASSES=CLASSES_
        self.num_pass_=numpass
        self.tprob_=terminal_prob
        self.class_pred_=class_pred
        self.decision_rules_=decision_rules
        self.value=class_pred
        self.n_node_samples=numpass
        self.threshold=pvalue_node
        self.class_names=CLASSES_
        self.ACC_=ACC_
        self.ACCx_=ACCx_
        self.response_var_=resp_
        self.significant_feature_weight_=sig_f_wt
        #self.n_outputs=n_outputs

#------------------------------------------
def upscale_(V,alpha=2,UL=255,LL=0):
    V=np.array(V)*alpha

    for i in np.arange(len(V)):
        if V[i] > UL:
            V[i]=UL

    return V


#------------------------------------------
def _color_brew(n,LIGHT=True,alpha=1.5):
    """Generate n colors with equally spaced hues.
    Parameters
    ----------
    n : int
        The number of colors required.
    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        if LIGHT:
            rgb=upscale_(rgb,alpha)

        color_list.append(np.array(rgb).astype(int))


    return color_list
#------------------------------------------

def normalize(arr_):
    s=np.sum(arr_)
    if s>0:
        return [ i/(0.0+s) for i in arr_]
    else:
        return arr_

def dround(DICT,SIG=2):
    P=DICT
    for key in P:
        P[key]=round(P[key],SIG)
    return P

#------------------------------------------
def xplot(NAMES,DATA,LABEL=None):
    thresholds = {
        .9: BIWhi,
        .6: BIBlu,
        .1: Blu,
    }
    graph = Pyasciigraph(
        line_length=50,
        min_graph_length=50,
        separator_length=5,
        multivalue=False,
        #graphsymbol='*'
    )
    for line in graph.graph(label=LABEL,
                            data=hcolor([ (NAMES[i],
                                           DATA[i])
                                          for i in np.arange(len(NAMES))],
                            thresholds)):
        print(line)

#------------------------------------------
def trim_(A):
    for key in A.keys():
        if key[0] == key[1]:
            A.pop(key, None)
    return

def trim__(A,keys):
    for key in A.keys():
        if key not in keys:
            A.pop(key, None)
    return

#------------------------------------------
def getlist_(class_vec_,CLS):
    clsv=[]
    for key in CLS:
        clsv.append(class_vec_[key])
    return clsv

#------------------------------------------
def sumlist_(c1,c2):
    clsv={}
    for key in c1.keys():
        if key not in c2.keys():
            print "ERR in sum"
            break
        else:
            clsv[key]=c1[key]+c2[key]
    return clsv

#------------------------------------------

def eqlist_(c1,c2):
    eq=True
    for key in c1.keys():
        if (key in c2.keys()) and eq:
            eq=(c1[key]==c2[key])
        else:
            return False
    return eq

#------------------------------------------

def getParent(children,nodeid_):
    for key in children.keys():
        if nodeid_ in children[key]:
            return key
    return -1

#------------------------------------------

def add_edge_conditions(edge_cond_,
                        path_to_root_,
                        cond_list_):
    for cnd in cond_list_:
        edge_cond_[(path_to_root_[-1],
                    path_to_root_[-2])]=cnd
        del path_to_root_[-1]
    return

#------------------------------------------

def rules_empty(RULES):
    if RULES[0][0][0]=='':
        return True
    else:
        return False

#------------------------------------------

def get_terminal_nodes_from_here(children__,
                                 thisnode__):
    if children__[thisnode__] == set():
        return [thisnode__]

    proc_list=[thisnode__]
    term_list=[]

    while proc_list:
        children_of_this=children__[proc_list[0]]
        for node in children_of_this:
            if children__[node]==set():
                term_list.append(node)
            else:
                proc_list.append(node)
        proc_list.pop(0)

    return term_list
 #------------------------------------------

def get_terminal_distinction_coeff(
    terminal_class_vector,
    terminal_node_list):

    coeff=1.0
    Max_class=set()
    for node in  terminal_node_list:
        thisvalue=0
        for class__,value__ in terminal_class_vector[node].items():
            if value__ > thisvalue:
                thisvalue=value__
                max_class=class__
        Max_class.add(max_class)

    if len(Max_class) > 1:
        return 1.0

    return 0.0

#------------------------------------------

def feature_importance(
    significant_dec_path,
    num_pass__,
    features_,
    tprob_significant_,
    children__,
    class_vector_):

    feature_imp={}
    #print "Significant decision paths: ",significant_dec_path
    for key__,path__ in significant_dec_path.items():
        old_num=0
        augmented_path=[]
        for this_node_ in path__:
            all_terminal_nodes=get_terminal_nodes_from_here(children__,this_node_)

            tcoeff=get_terminal_distinction_coeff(class_vector_,all_terminal_nodes)

            gfrac=(num_pass__[this_node_]-old_num)/(0.01+num_pass__[this_node_])
            ggfrac=4*gfrac*(1-gfrac)
            old_num=num_pass__[this_node_]

            if this_node_ != key__:
                feature_imp[features_[this_node_]] \
                    =feature_imp.get(features_[this_node_],0) \
                    +tcoeff*ggfrac*tprob_significant_[key__]

    return feature_imp


#------------------------------------------

def visTree(
    MODEL,
    PR,
    PLOT=True,
    VERBOSE=False,
    ACC=None,
    ACCx=None,
    RESP_=None,
    PROB_MIN=0.1):

    RLS=rls(MODEL)
    CLASSES=PR.columns.values[1:-2]
    ID=ndid(MODEL,terminal=True)
    
    CFRQ=[normalize([PR[PR.nodeid==i][PR.orig_response==j].index.size
           for j in CLASSES])
          for i in ID]
    tprob=getterminalprob(MODEL,PR)
    tprob_significant={key__:tval for key__,tval in tprob.items()
                       if tval >= PROB_MIN}
    #if DEBUG__:
    #    print "########## ", tprob_significant

    RLS_=[[ i.split('%in%')
            for i in j.split('&')]
          for j in rls(MODEL) ]

    count__=0
    node_seq_from_rules_=[]

    for rl in RLS_:
        var_node_=[]
        for edg in rl:
            var_node_.append(edg[0].strip())
        var_node_.append(str(ID[count__]))
        node_seq_from_rules_.append(var_node_)
        count__=count__+1

    features={}
    leaf_={}
    children_left={}
    children_right={}
    children={}
    class_vector_={}
    error_={}
    num_pass_={}
    pvalue_={}

    for node in ndid(MODEL):
        A=sapply(nodeapply(assimpleparty(MODEL), ids = node,
                           FUN=infonode),criteria="distribution")
        class_vector_[node]=dict(zip(A.names[0],list(A)))

        A=sapply(nodeapply(assimpleparty(MODEL),ids = node,
                           FUN=infonode),criteria="error")
        A=dict(zip(A.names[0].replace(str(node),'E'),list(A)))
        error_[node]=A['E']

        A=sapply(nodeapply(assimpleparty(MODEL), ids = node,
                           FUN=infonode),criteria="n")
        A=dict(zip(A.names[0].replace(str(node),'N'),list(A)))
        num_pass_[node]=A['N']

        if not (node in ID):
            tmpfilename_=id_generator(16)
            wrtcsv(sapply(nodeapply(assimpleparty(MODEL),
                                    ids = node,FUN=infonode),
                          criteria="p.value"),tmpfilename_)
            A=pd.read_csv(tmpfilename_)
            pvalue=A.values[0][1]
            len_=len(str(node))+1
            nodevar=A.values[0][0][len_:]
            features[node]=nodevar
            pvalue_[node]=pvalue
            os.remove(tmpfilename_)
            leaf_[node]=False
            children[node]=set()
        else:
            features[node]=str(node)
            leaf_[node]=True
            children[node]=set()


    childnodes=set()
    for thisnode in ndid(MODEL):
        if not (thisnode in ID):
            for node1 in ndid(MODEL):
                if node1 in childnodes:
                    continue
                for node2 in ndid(MODEL):
                    if node2 in childnodes:
                        continue
                    if (node1 != node2) and eqlist_(class_vector_[thisnode],
                                                    sumlist_(class_vector_[node1],
                                                             class_vector_[node2])):
                        CHECK_A_=False
                        CHECK_B_=False
                        for seq__ in node_seq_from_rules_:
                            for index_ in np.arange(len(seq__)-1):
                                if (seq__[index_] == features[thisnode]) and  (seq__[index_+1] == features[node1]):
                                    CHECK_A_=True
                                    break
                        for seq__ in node_seq_from_rules_:
                            for index_ in np.arange(len(seq__)-1):
                                if (seq__[index_] == features[thisnode]) and  (seq__[index_+1] == features[node2]):
                                    CHECK_B_=True
                                    break
                        if CHECK_A_ & CHECK_B_:
                            if DEBUG__:
                                print thisnode, "children ", node1,node2
                                print num_pass_[thisnode], "n1 :", num_pass_[node1], "n2 :",num_pass_[node2]

                            children[thisnode]=set([node1,node2])
                            childnodes.add(node1)
                            childnodes.add(node2)
                            break

        if children[thisnode]!=set():
            children_left[thisnode]=list(children[thisnode])[0]
            children_right[thisnode]=list(children[thisnode])[1]
        else:
            children_left[thisnode]=None
            children_right[thisnode]=None

    edge_cond_={}
    class_pred={}
    decision_rules={}
    count=0

    if DEBUG__:
        print RLS
        print RLS_

    if rules_empty(RLS_):
            return None

    sig_dec_paths_={}
    for rl in RLS_:
        cond_list_=[]
        _prev_key_=0

        if DEBUG__:
            print 'rl: ', rl

        for edg in rl:
            if DEBUG___:
                print 'edg: ', edg

            var_node_=edg[0].strip()
            var_values_=list(set(edg[1].strip().replace('c(','')
                                 .replace(')','').replace('"','')
                                 .split(", "))-set(['NA']))
            cond_list_.append(var_values_)
            _prev_val_=var_values_
        path_to_root=[ID[count]]


        if DEBUG__:
            print 'children: ',children

        while path_to_root[-1] > 1:
            if DEBUG__:
                print "--> path_to_root:", path_to_root
            path_to_root.append(getParent(children,path_to_root[-1]))

        if DEBUG__:
            print "===> path_to_root:", path_to_root

        if ID[count] in tprob_significant.keys():
            sig_dec_paths_[ID[count]]=np.array(path_to_root)

        decision_rules[ID[count]]=rl
        add_edge_conditions(edge_cond_,path_to_root,cond_list_)

    #    if PLOT:
    #        xplot(CLASSES,100*CFRQ[count])
        class_pred[ID[count]]=CFRQ[count]
        count=count+1

    decision_rules__={}
    for tnd in np.arange(len(ID)):
        decision_rules__[ID[tnd]]=RLS[tnd]

    if DEBUG__:
        print "######## sdr:",sig_dec_paths_

    sig_feature_weight=feature_importance(sig_dec_paths_,
                                          num_pass_,
                                          features,
                                          tprob_significant,
                                          children,
                                          class_vector_)
    TR=tree_(ndid(MODEL),feature=features,
             leaf_=leaf_,
             children=children,
             children_left=children_left,
             children_right=children_right,
             edge_cond_=edge_cond_,
             error_node_=error_,
             pvalue_node=pvalue_,
             CLASSES_=CLASSES,
             numpass=num_pass_,
             terminal_prob=getterminalprob(MODEL,PR),
             class_pred=class_vector_,decision_rules=decision_rules__,
             ACC_=ACC,ACCx_=ACCx,resp_=RESP_,sig_f_wt=sig_feature_weight)

    if VERBOSE:
        sys.stdout.write(WHITE)
        print
        print "========= DECISION TREE ================  "
        sys.stdout.write(BOLD)
        print "#Feature name--: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.feature)
        sys.stdout.write(BOLD)
        print "#Leaf----------: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.TREE_LEAF)
        sys.stdout.write(BOLD)
        print "#Children------: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.children)
        sys.stdout.write(BOLD)
        print "#Edge condition: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.edge_cond_)
        sys.stdout.write(BOLD)
        print "#Error(%)------: "
        sys.stdout.write(WHITE)
        pp.pprint(dround(TR.error))
        sys.stdout.write(BOLD)
        print "#Pvalue--------: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.pvalue)
        sys.stdout.write(BOLD)
        print "#Classes-------: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.CLASSES)
        sys.stdout.write(BOLD)
        print "#Num_passes----: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.num_pass_)
        sys.stdout.write(BOLD)
        print "#Terminal_prob-: "
        sys.stdout.write(WHITE)
        pp.pprint(dround(TR.tprob_))
        sys.stdout.write(BOLD)
        print "#Class_pred----: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.class_pred_)
        sys.stdout.write(BOLD)
        print "#RULES---------: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.decision_rules_)
        print "#SIGNIFICANT_FEATURES---------: "
        sys.stdout.write(WHITE)
        pp.pprint(TR.significant_feature_weight_)
        print "----------------------------------"
        print

        sys.stdout.write(RESET)

    return TR

#------------------------------------------

def nameclean(str_):
    str_.replace('-','x').replace('+','p').replace('*','s').replace('.','d')
    # if unicode(str_).isdigit():
    str_='P'+str_

    return str_

def setdataframe(
    file1,
    outname="",
    delete_=[],
    include_=[],
    select_col=False,
    rand_col_sel=10,
    response_var=[],
    balance=False,
    zerodel=[],
    VERBOSE=False):
    
    MINCLASSNUM=70
    D1=pd.read_csv(file1,delimiter=",",index_col=None,
                   engine='python')
    D1.columns = map(nameclean,D1.columns.values)
    X_train=D1.values
    datatrain = pd.DataFrame(X_train,columns=D1.columns)#.dropna('columns')

    if balance:
        DD={}
        DDlen=[]
        if response_var is not None:
            valset=set(datatrain[response_var[0]].values)
            for i in valset:
                dd=datatrain[datatrain[response_var[0]]==i]
                DD[len(dd.index.values)]=dd
                DDlen.append(len(dd.index.values))
            DDlen=np.sort(np.array(DDlen))

            DDlen=DDlen[DDlen>MINCLASSNUM]
            minlen=DDlen[0]

            DD__=[DD[minlen]]
            for i in DDlen[1:]:
                DD__.append(DD[i].sample(n=minlen))
            datatrain=pd.concat(DD__)
            sys.stdout.write(PURPLE)
            print "Balancing complete: ",DDlen, len(datatrain.index.values)
            sys.stdout.write(WHITE)


    if len(include_)>0:
        delete_all_but_include_=[item for item in datatrain.columns.values if item not in include_]
        datatrain.drop(delete_all_but_include_,axis=1,inplace=True)

    if select_col:
        datatrain_tmp=datatrain.sample(n=rand_col_sel,axis=1)
        for r in response_var:
            if r not in datatrain_tmp.columns:
                datatrain_tmp[r]=datatrain[r]
        datatrain=datatrain_tmp

    if len(delete_)>0:
        datatrain.drop(delete_,axis=1,inplace=True)

    for val in zerodel:
        datatrain=datatrain[datatrain.eval(response_var[0]) != val]

    datatrain=datatrain.reset_index().drop('index', axis=1)

    if VERBOSE:
        print "(samples,features): ", datatrain.shape, "deleted: ", delete_

    if outname != "":
        datatrain.to_csv(outname,index=False)
    return datatrain

#------------------------------------------

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return 'TMP' + ''.join(random.choice(chars) for _ in range(size))

#------------------------------------------

def getresponseframe(DATA,MODEL,RESPONSE_,olddata=False):
    """Using a trained ctree model, generate a response frame.

    Args:
        DATA: input data of the sequences in csv format
        MODEL: ctree from rpy2
        RESPONSE_: the location of the target
        olddata: if True, DATA is train data. Otherwise, DATA is test data.

    Returns:
        PR: dataframe containing the node id, 
            probability of each label, 
            predicted response, 
            and original response
        ACC: accuracy of the prediction
        cf: confusion matrix
    """

    tmpfilename=id_generator(16)
    if(olddata):
        wrtcsv(prd(MODEL,type="prob"),tmpfilename)
    else:
        wrtcsv(prd(MODEL,newdata=DATA,type="prob"),tmpfilename)

    PR=pd.read_csv(tmpfilename)
    PR.rename(columns={PR.columns[0]:'nodeid'},inplace=True)
    if(olddata):
        wrtcsv(prd(MODEL,type="response"),tmpfilename)
    else:
        wrtcsv(prd(MODEL,newdata=DATA,type="response"),tmpfilename)
    Pr_tmp=pd.read_csv(tmpfilename)
    PR["pred_response"]=Pr_tmp.x
    PR["orig_response"]=DATA[RESPONSE_]

    PRs=PR[['pred_response','orig_response']]
    A={}
    for i in set(PRs.pred_response.values):
        for j in set(PRs.orig_response.values):
            A[(i,j)]=PRs[(PRs.pred_response==i)&(PRs.orig_response==j)].index.size
    data = map(list, zip(*A.keys())) + [A.values()]
    cf = pd.DataFrame(zip(*data)).set_index([0, 1])[2].unstack()
    cf=cf.combine_first(cf.T).fillna(0)
    cf.index.name='pred.'
    cf=cf.astype(int)

    ACC=1- (PR[PR.pred_response
               !=PR.orig_response].index.size/(0.0+PR.index.size))
    os.remove(tmpfilename)
    return PR,ACC,cf

#------------------------------------------

def getresponseframe_RF(DATA,MODEL,RESPONSE_,olddata=False):
    tmpfilename=id_generator(16)
    if(olddata):
        wrtcsv(prd(MODEL,type="prob"),tmpfilename)
    else:
        wrtcsv(prd(MODEL,DATA,type="prob"),tmpfilename)

    PR=pd.read_csv(tmpfilename)
    PR.rename(columns={PR.columns[0]:'nodeid'},inplace=True)
    if(olddata):
        wrtcsv(prd(MODEL,type="response"),tmpfilename)
    else:
        wrtcsv(prd(MODEL,DATA,type="response"),tmpfilename)
    Pr_tmp=pd.read_csv(tmpfilename)
    PR["pred_response"]=Pr_tmp.x
    PR["orig_response"]=DATA[RESPONSE_]

    PRs=PR[['pred_response','orig_response']]
    A={}
    for i in set(PRs.pred_response.values):
        for j in set(PRs.orig_response.values):
            A[(i,j)]=PRs[(PRs.pred_response==i)&(PRs.orig_response==j)].index.size
    data = map(list, zip(*A.keys())) + [A.values()]
    cf = pd.DataFrame(zip(*data)).set_index([0, 1])[2].unstack()
    cf=cf.combine_first(cf.T).fillna(0)
    cf.index.name='pred.'
    cf=cf.astype(int)

    ACC=1- (PR[PR.pred_response
               !=PR.orig_response].index.size/(0.0+PR.index.size))
    os.remove(tmpfilename)
    return PR,ACC,cf

#------------------------------------------

def getDataFrame(VAR,COLNAME=""):
    tmpfilename=id_generator(16)
    wrtcsv(VAR,tmpfilename)
    VAR1=pd.read_csv(tmpfilename)
    if COLNAME!="":
        VAR1.rename(columns={VAR1.columns[0]:COLNAME},inplace=True)
    os.remove(tmpfilename)
    return VAR1
#------------------------------------------


def getterminalprob(MODEL,PR_):
    ID=ndid(MODEL,terminal=True)
    freq_=[PR_[PR_.nodeid==i].index.size for i in ID]
    s=(0.0+np.sum(freq_))
    FRQ={}
    freq_=[ i/s for i in freq_]
    for i in np.arange(len(ID)):
        FRQ[ID[i]]=freq_[i]
    return FRQ
#------------------------------------------

def plotfi(MODELimp):
    sys.stdout.write(BOLD)
    thresholds = {
        9: BIWhi,
        8: BIBlu,
        6: Blu,
        4: Bla,
        2: BIBla,
    }
    graph = Pyasciigraph(
        line_length=70,
        min_graph_length=50,
        separator_length=10,
        multivalue=False,
        #graphsymbol='*'
    )
    for line in graph.graph(label='Feature Importance (GINI)',
                            data=hcolor([ (MODELimp.feature[i],
                                           MODELimp.MeanDecreaseGini[i])
                                          for i in MODELimp.head(10).index],
                                        thresholds)):
        print(line)
    sys.stdout.write(RESET)
    return

#------------------------------

#------------------------------
#--------   R CODE  ----------
#------------------------------
pk = importr('partykit')
#pk = importr('party')
stats = importr('stats')
base = importr('base')
rf=importr('randomForest')

ctree=robjects.r('ctree')
#if VARIMP:
randomForest_=robjects.r('randomForest')
rimp=robjects.r('importance')
randomForest=robjects.r('cforest')
#rimp_=robjects.r('varimp')
rpred=robjects.r('predict')
pltr=robjects.r('plot')
prnt=robjects.r('print')
ppp=robjects.r('as.simpleparty')
rls=robjects.r('partykit:::.list.rules.party')
prd=robjects.r('predict')
ndid=robjects.r('nodeids')
wrtcsv=robjects.r('write.csv')
readcsv = robjects.r(
    '''
    read_csv <- function(r) {
        data <- read.csv(r, na.strings=c("NA","NaN", ""))
        return(data)
    }
    ''')

nodeapply=robjects.r('nodeapply')
assimpleparty=robjects.r('as.simpleparty')
infonode=robjects.r('info_node')
sapply_=robjects.r('sapply')
#------------------------------

def sapply(node,arg2="[[",criteria="error"):
    return sapply_(node,arg2,criteria)


#------------------------------------------------------------

def Xctree(
    RESPONSE__,
    datatrain__,
    datatest__=None,
    VERBOSE=False,
    TREE_EXPORT=True):
    """Train the conditional tree.
    
    NOTE: for some reason, sometimes getresponseframe doesn't work

    Args:
        RESPONSE__ (str): response name
        datatrain__ (pandas df): train data
        datatest__ (pandas df): evaluation data
        VERBOSE (bool): whether to print out extra info
        TREE_EXPORT (bool): whether to export the tree to a file

    Returns:

    """

    Prx__  = None
    ACCx__ = None
    CFx__ = None
    
    # We need to read the data file using R. 
    # If we don't, rpy2 will force an invalid conversion from pandas dataframe
    # to R dataframe.
    tmpfile = id_generator(16)
    datatrain__.to_csv(
        tmpfile,
        sep=',',
        index=None)
    datatrain__2 = readcsv(tmpfile)
    os.remove(tmpfile)

    fmla__ = Formula(RESPONSE__+' ~ .')
    CT__ = ctree(
        fmla__,
        data=datatrain__2)
    import pdb; pdb.set_trace()
    Pr__, ACC__, CF__= getresponseframe(
        datatrain__,
        CT__,
        RESPONSE__,
        olddata=True)

    if datatest__ is not None:
        try:
            Prx__, ACCx__, CFx__ = getresponseframe(
                datatest__,
                CT__,
                RESPONSE__)

        # some weird error regarding a column name not found
        except rinterface.RRuntimeError:
            Prx__ = ACCx__ = CFx__ = None
            
    TR__ = visTree(
        CT__,
        Pr__,
        PLOT=False,
        VERBOSE=VERBOSE,
        ACC=ACC__,
        ACCx=ACCx__,
        RESP_=RESPONSE__)

    if TR__ is not None:
        if TREE_EXPORT:
            tree_export(TR__,TYPE='polyline',EXEC=True)

    return CT__, Pr__, ACC__, CF__, Prx__, ACCx__, CFx__, TR__

#------------------------------------------------------------


def tree_export(
    TR,
    outfilename='out.dot',
    leaves_parallel=True,
    rounded=True,
    filled=True,
    TYPE='Curved',
    BGCOLOR='transparent',
    legend=True,
    LIGHT=1,
    LABELCOL='deepskyblue4',
    TEXTCOL='black',
    EDGECOLOR='gray',
    EXEC=True):

    LABELTYPE='label'
    #if TYPE=='ortho':
    #    LABELTYPE='xlabel'
    out_file = open(outfilename, "w")

    out_file.write('digraph Tree {\n')
            # Specify node aesthetics
    out_file.write('node [shape=box')
    rounded_filled = []
    if filled:
        rounded_filled.append('filled')
    if rounded:
        rounded_filled.append('rounded')
    if len(rounded_filled) > 0:
        out_file.write(', style="%s", color="%s"'
                       % (", ".join(rounded_filled),TEXTCOL))
    if rounded:
        out_file.write(', fontname=helvetica')
    out_file.write('] ;\n')

    # Specify graph & edge aesthetics
    if leaves_parallel:
        out_file.write('graph [ranksep=equally, splines=%s, bgcolor=%s, dpi=600] ;\n' % (TYPE,BGCOLOR))
    else:
        out_file.write('graph [splines=%s, bgcolor=%s, dpi=600] ;\n' % (TYPE,BGCOLOR))

    if rounded:
        out_file.write('edge [fontname=helvetica,color=%s] ;\n' % EDGECOLOR)
    if rotate:
        out_file.write('rankdir=LR ;\n')

    COLORS=_color_brew(len(TR.CLASSES),LIGHT=True,alpha=LIGHT)
    COLORS_=['#'+''.join(map(chr, triplet)).encode('hex')
             for triplet in COLORS]

    node_color={}
    for node_id in TR.class_pred_.keys():
        cls_prd_n=normalize(np.array(getlist_(TR.class_pred_[node_id],TR.CLASSES)))
        COL=np.zeros(3)
        for j in np.arange(len(cls_prd_n)):
            COL=np.sum([COL,cls_prd_n[j]*np.array(COLORS[j])],axis=0)

        node_color[node_id]='#'+''.join(map(chr,
                                np.array(COL).astype(int))).encode('hex')

    node_str={}
    for node_id in TR.feature.keys():
        STR=""
        if TR.TREE_LEAF[node_id]:
            STR=STR+str(TR.CLASSES[np.argmax(np.array(getlist_(TR.class_pred_[node_id],TR.CLASSES)))])
            STR=STR+"\nProb: "+str(round(TR.tprob_[node_id],2))
            STR=STR+"\nErr: "+str(round(TR.error[node_id],2))+"%"
        else:
            STR=STR+TR.feature[node_id]
            STR=STR+"\n\npval: "+str('%.2E' % decimal.Decimal(TR.pvalue[node_id]))
        node_str[node_id]=STR

    if legend:
        LEGENDSTR="Response : "+TR.response_var_+ "\n"
        LEGENDSTR=LEGENDSTR+"Classes: "+'|'.join(TR.CLASSES)+"\n"
        if TR.ACC_ is not None:
            LEGENDSTR=LEGENDSTR+"In ACC: "+str(round(TR.ACC_,2))+"\n"
        if TR.ACCx_ is not None:
            LEGENDSTR=LEGENDSTR+"Out ACC: "+str(round(TR.ACCx_,2))+"\n"
        out_file.write('LEGEND [label="%s",shape=note,align=left,style=filled,fillcolor="slategray",fontcolor="white",fontsize=10];' % LEGENDSTR)

    for node_id in TR.feature.keys():
        out_file.write('%d [label="%s"' % (node_id , node_str[node_id]))
        if filled:
            out_file.write(', fillcolor="%s",fontcolor="%s"' % (node_color[node_id],TEXTCOL))
        out_file.write('] ;\n')

    for parent in TR.children.keys():
        if not TR.TREE_LEAF[parent]:
            out_file.write('%d -> %d [%s="%s",fontcolor=%s' % (parent,
                                                     TR.children_left[parent],LABELTYPE,
                                                               ''.join(TR.edge_cond_[(parent,TR.children_left[parent])]),LABELCOL))
            out_file.write('] ;\n')
            out_file.write('%d -> %d [%s="%s",fontcolor=%s' % (parent,
                                                     TR.children_right[parent],LABELTYPE,
                                                     ''.join(TR.edge_cond_[(parent,TR.children_right[parent])]),LABELCOL))
            out_file.write('] ;\n')

    if leaves_parallel:
        STR="{rank = same; "
        for node_id in TR.feature.keys():
            if TR.TREE_LEAF[node_id]:
                STR=STR+str(node_id)+";"
        STR=STR+"}"
        out_file.write(STR)

    if legend:
        out_file.write('{rank = same; LEGEND;1;}')

    out_file.write("}")

    if EXEC:
        path, basename = os.path.split(outfilename)
        outfilename_ = os.path.join(path, 'TREE_' + basename.replace('.dot', '.png'))
        subprocess.Popen(["dot", '-Tpng', outfilename, '-o', outfilename_])
    return




#---------------------------------------------------------------
#---------------------------------------------------------------
def randomForestX(
    RESPONSE__,
    datatrain__,
    datatest__=None,
    NUMTREE=300,
    CORES=1,
    VERBOSE=False,
    VARIMP=True,
    PLOT=True):

    PrxRF=None
    ACCxRF=None
    CFxRF=None
    EFI=None
    RFimp=None

    if VERBOSE:
        print "Growing forest..(using ",CORES," cores)"

    fmla__ = Formula(RESPONSE__+' ~ .')
    RF=randomForest(fmla__,data=datatrain__,
                    ntree=NUMTREE,
                    trace=True,
                    cores=CORES)
    if VARIMP:
        RF__=randomForest_(fmla__,data=datatrain__,ntree=NUMTREE)
        RFimp=getDataFrame(rimp(RF__),
                           'feature').sort_values('MeanDecreaseGini',
                                                  ascending=False)
        RFimp.to_csv('imp.czv')
        if PLOT:
            plotfi(RFimp)
        else:
            print RFimp.head(20)

        PrRF,ACCRF,CF_=getresponseframe_RF(datatrain__,
                                           RF__,
                                           RESPONSE__,
                                           olddata=True)
        sys.stdout.write(WHITE)
        print
        print "ACC (in  sample, randomForest package): ",ACCRF
        sys.stdout.write(RESET)
        EFI=stat.entropy(RFimp.MeanDecreaseGini.values,base=2)
        print
        print "Entropy of Feature Importance: ",EFI


    PrRF,ACCRF,CFRF=getresponseframe_RF(datatrain__,
                                        RF,RESPONSE__,
                                        olddata=True)
    if datatest__ is not None:
        PrxRF,ACCxRF,CFxRF=getresponseframe_RF(datatest__,
                                                  RF,
                                                  RESPONSE__)

    sys.stdout.write(RED)
    print
    print "ACC (in  sample, Random Forest): ",ACCRF
    if datatest__ is not None:
        print
        print "ACC (out sample, Random Forest): ",ACCxRF
        sys.stdout.write(CYAN)
        print "Out of Sample Confusion Matrix:"
        print CFxRF
        sys.stdout.write(RESET)


    return RF,PrRF,ACCRF,CFRF,PrxRF,ACCxRF,CFxRF,RFimp,EFI
#---------------------------------------------------------------



def dictprod(dict_,a=1.0):
    '''
        given a dict of probability distributions 
        represented as such: {'key1': val1, ... ,'keyn':valn}
        multiply all values with the second argument `a`
    '''
    return {key:value*a for (key,value) in dict_.iteritems()}
 
def normalizedict(dict_):
    '''
        given a dict represented as such: {'key1': val1, ... ,'keyn':valn}
        scale all values such that they sum to 1.0    
    '''
    s=0.0
    for key in dict_.keys():
        s=s+dict_[key]
    return {key:(value/s) for (key,value) in dict_.iteritems()}

def mergedistributions(dist_):
    '''
        given a dict of dicts, each represented as such: 
        {'key1': val1, ... ,'keyn':valn}
        we retun a combined dict, where values corresponding  
        to key1 is the average over 
        all component dicts
    '''
    num=len(dist_.keys())
    key_list=[]
    for key in dist_.keys():
        key_list=np.append(key_list,dist_[key].keys())
        
    D={}
    for key in key_list:
        D[key]=0.0
        
    for count in dist_.keys():
        for key_ in dist_[count].keys():
            if key_ in dist_[count]:
                D[key_]=D[key_]+dist_[count][key_]
    return {key:value/(num+0.0) for (key,value) in D.iteritems() }

def getMergedDistribution(tree,cond={}):
    '''
        get distribution over keys given particular
        constriants (cond) on the decision tree
        
        Arguments:
        
        tree: decision tree returned by mlx.py
        cond: conditions that specify constraints
              on the decision tree
        
    '''
    node_id_map={feature_name:np.array([], dtype=int)
                 for (i,feature_name) in tree.feature.iteritems()}
    for (i,feature_name) in tree.feature.iteritems():
        node_id_map[feature_name]=np.append(node_id_map[feature_name],int(i))
    
    if DEBUG:
        print(node_id_map)
    #propagate to find current nodes
    children={i:set() for i in cond.keys()}
    for feature_name in cond.keys():
        for node_id in tree.feature:
            if tree.feature[node_id] == feature_name:
                children[feature_name]=children[feature_name].union(tree.children[node_id])
    if DEBUG:
        print(children)

    current_active_nodes=np.array([],int)
    for feature_name in cond.keys():
        for child in children[feature_name]:
            for parent in node_id_map[feature_name]:
                if (parent,child) in tree.edge_cond_:
                    for edge_var in cond[feature_name]:
                        if edge_var in tree.edge_cond_[(parent,child)]:
                            if DEBUG:
                                print(parent,child,"::",tree.edge_cond_[(parent,child)])
                            current_active_nodes=np.append(current_active_nodes,child)
    
    S=0.0
    if current_active_nodes.size == 0:
        current_active_nodes=np.array([1],int)
    for i in current_active_nodes:
        S=S+tree.num_pass_[i]
        
    indexed_dist={i:dictprod(tree.class_pred_[i],tree.num_pass_[i]/S)
                  for i in current_active_nodes}
    dist_=normalizedict(mergedistributions(indexed_dist))
        
    if DEBUG:
        print(children)
        print(current_active_nodes)
        print("ID",indexed_dist)
        print("MD",mergedistributions(indexed_dist))
        print("ND",normalizedict(mergedistributions(indexed_dist)))
        
    return dist_  
    
def sampleTree(tree, cond={}, sample='mle', DIST=False, NUMSAMPLE=10):
    '''Draw sample from the decision tree.

    specified in the format that 
    mlx.py returns
    
    Arguments:
    
    1. cond: dict of the format {'name': value, 'name1': value1,...}
                specifies the constraints on the decision tree.
                example: {'RBM34':'C','SOX2': 'A'}
    
    Note--> we can use arbitrary cond argument, irrespective of if the
    names are in the decision tree at all or not. Also, we can use 
    an empty cond dict, which corresponds to the unconstrained tree.
    In all these cases, it makes sense to ask what is the distribution on the 
    keys that the decision tree outputs, and we attempt to compute that.
    
    NOTE: cond with floats as values will be filtered out.

    2. sample: 'mle'|'random' 
                if 'mle' then return the value with maximum probability.
                if 'random' then makes random choice NUMSAMPLE times 
                and returns the result.
    
    3. DIST: TRUE|FALSE
                if TRUE returns the distribution from the tree 
                after applying the constraints
    '''

    items = list(cond.keys())
    for item in items:
        if not item.startswith('P'):
            raise ValueError('The response name must start with P!')

    # filter out values with floats
    # convert booleans to strings
    cond_ = {}
    for k, v in cond.items():
        if type(v) is float:
            pass
        elif type(v) is bool:
            cond_[k] = str(v)
        else:
            cond_[k] = v

    dist_ = getMergedDistribution(tree,cond=cond_)
    if sample is 'mle':
        sample = max(dist_.iteritems(), key=operator.itemgetter(1))[0]
    elif sample is 'random':
        probs = dist_.values()
        keys =  dist_.keys()
        sample = np.random.choice(keys,NUMSAMPLE, replace=True, p=probs)
    else:
        raise ValueError('Not a correct sampling method.')
    if DIST:
        return sample, dist_
    return sample

def getFmap(PATH_TO_TREES):
    F={}
    TREE={}
    TREES=glob.glob(PATH_TO_TREES)
    for filename in TREES:
        with open(filename,'rb') as f:
            TR = pickle.load(f)
        f.close()
        index=os.path.splitext(os.path.basename(filename))[0].split('_')[-1]
        #print index
        F[index]=[]
        TREE[index]=TR
        for key,value in TR.feature.iteritems():
            if not TR.TREE_LEAF[key]:
                F[index]=np.append(F[index],value)
    return F,TREE

def getPerturbation(seq,PATH_TO_TREES):
    F,TREES=getFmap(PATH_TO_TREES)
    P={}
    for KEY in F.keys():
        I=[int(x.replace('P','')) for x in F[KEY]]
        DICT_={'P'+str(i):seq[i] for i in I}
        D=sampleTree(TREES[KEY],DICT_,sample='random',DIST=True)[1]
        
        P[KEY]=[D[x] for x in sorted(D.keys()) ]
    return P

def klscore(p1,p2):
    
    if any(np.array(p2)<=0):
        return np.nan
    
    return np.array([p1[i]*np.log2(p1[i]/p2[i]) for i in range(len(p1))]).sum()

def jsdiv(p1,p2,smooth=True):
    
    
    p1=np.array(p1)
    p2=np.array(p2)

    if(smooth):
        p1=(p1+0.0001)/1.0001
        p2=(p2+0.0001)/1.0001
    
    p=0.5*(p1+p2)
    return 0.5*(klscore(p1,p)+klscore(p2,p))


def qDistance(seq0,seq1,PATH_TO_TREES):
    '''
     computing genomic distance using qnets
    '''
    P0=getPerturbation(seq0,PATH_TO_TREES)
    P1=getPerturbation(seq1,PATH_TO_TREES)
    S=0.0
    nCount=0

    # @TIMMY parallelize the following loop
    for key0 in P0.keys():
        if key0 in P1.keys():
            S=S+jsdiv(P0[key0],P1[key0])
            nCount=nCount+1
    if nCount == 0:
        nCount=1
    return S/(nCount+0.0)


def load_trees(tree_dir, return_items=False):
    """Load the trees into a dictionary from a directory.

    Args:
        tree_dir (str): directory to the store trees
        return_items (bool): whether to return responses or not

    Returns:
        dictionary mapping item name to the corresponding tree
    """

    if not tree_dir.endswith('/'):
        raise ValueError("The tree directory must end with: /")

    pickled_tree_files = glob.glob(tree_dir + '*.pkl')
    pickled_tree_files.sort()

    # items are the file names gathered from the pickle file name
    items = [file_.split('/')[-1].split('.')[0] \
        for file_ in pickled_tree_files]

    trees = {}
    for i, pickled_tree_file in enumerate(pickled_tree_files):
        with open(pickled_tree_file, 'rb') as f:
            tree = pickle.load(f)
        trees[items[i]] = tree

    if return_items:
        return trees, items
    else:
        return trees


def dissonance(dist, response):
    """Compute the dissonance for item i.

    If the response is NaN, then the probability chosen is the maximum
    from the distribution.

    Args:
        dist: a dictionary that map label name to probability.
        response: the actual response for the distribution

    Returns:
        dissonance scalar for item i
        1 if the probability that we got is 0
    """

    if type(response) is bool:
        response = str(response)

    if isnan(response):
        prob = max(dist.values())
    else:
        prob = dist[response]

    if prob == 0.0:
        return 1
    else:
        v = 1 - 2 ** (prob * np.log2(prob))
        return v


def dissonanceVector(dists, responses):
    """Calculate the dissonance vector for a single instance.

    Args:
        dists: list of dictionary that map label name to probability.
        responses: list of responses for the distributions

    Returns:
        list of dissonances
    """

    if len(dists) != len(responses):
        raise ValueError('Number of distributions must match number of responses.')

    vs = []
    for i, dist in enumerate(dists):
        response = responses[i]
        v = dissonance(dist, response)
        vs.append(v)

    return vs



def sampleDissonanceVector(df, tree_dir, save_file=None):
    """For each sample in a dataframe, find the dissonance vector.

    Args:
        df (pandas df): dataframe containing the data
        tree_dir (str): directory that the trees were saved
        save_file (str): file to save the dissonance vectors

    Returns:
        2d numpy array of size 
            (number of samples, number of pickle files in tree_dir)
    """

    trees, items = load_trees(tree_dir, return_items=True)

    samples = df.shape[0]
    col_names = df.columns

    # store a lists of dissonance vectors
    all_vecs = []

    # iterate over the rows of the dataframe
    for row_index in range(samples):

        # cond_dict maps item names to actual values from the data
        cond_dict = {}
        for col_name in col_names:
            cond_dict[col_name] = df[col_name][row_index]

        # labels are a list of actual values for each item
        labels = []

        # dists is a list of dictionaries
        # each map possible labels to probabilities of that label
        dists = []
        
        # iterate over the items to find the distribution
        for item in items:

            labels.append(df[item][row_index])
            tree = trees[item]
            distrib_dict = copy.deepcopy(cond_dict)
            del distrib_dict[item]
            result, dist_ = sampleTree(
                tree, 
                cond=distrib_dict,
                DIST=True,
                sample='random')

            dists.append(dist_)

        # convert bools to strings
        labels_ = []
        for label in labels:
            if type(label) is bool:
                label = str(label)
            labels_.append(label)

        v = dissonanceVector(dists, labels_)
        all_vecs.append(v)

    all_vecs = np.array(all_vecs)

    # save the dissonance vectors to file
    if save_file is not None:
        df = pd.DataFrame(
            data=all_vecs, 
            columns=items)
        df.to_csv(
            save_file, 
            index=None)

    return all_vecs, items


def alpha_i(
    orig_dissonance, 
    cond_dict, 
    item_i,
    trees, 
    items_to_all_responses,
    num_samples=100):
    """Compute the alpha parameter for item i.
    
    Alpha measures the difficulty of reducing dissonance when
    responses other than i are perturbed.

    Alpha is calculated by:
        1. fixing the response i
        2. randomly selecting an item j that is not i
        3. randomly sampling a possible response from j
        4. replacing the original response of j with the sampled j response
        5. recomputing the dissonance of i
        6. repeating step 1 to 5 n times
        7. calculating the proportion of times the recomputed dissonance is less than the original dissonance

    Args:
        orig_dissonance: original dissonance of item i
        cond_dict: dict that maps item names to actual responses
        item_i: name of item i
        trees: dictionary mapping each item to its corresponding tree
        num_samples: number of times of resampling
        items_to_all_responses: map item to all possible responses for that item

    Returns:
        proportion of sampled dissonance less than original dissonance
        nan if the response of item i is nan
    """

    # all_items = list(trees.keys())
    # items_except_i = [x for x in all_items if x != item_i]
    tree_i = trees[item_i]
    response_i = cond_dict[item_i]

    sampled_dissonances = []

    # j_choices = set(tree_i.feature.values())
    # j_choices = j_choices.intersection(set(items_except_i))
    # j_choices = list(j_choices)

    j_choices = tree_i.significant_feature_weight_.keys()

    j_choices = [j_choice for j_choice in j_choices \
        if len(items_to_all_responses[j_choice]) != 0 ]

    # I don't think this scenario will happen
    if len(j_choices) == 0:
        return float('nan')


    for _ in range(num_samples):
        # tree_j = trees[item_j]
        # sample_j = random.choice(tree_j.CLASSES)
        # sample_j = sampleTree(
        #     tree_j,
        #     cond={},
        #     sample='random',
        #     DIST=False,
        #     NUMSAMPLE=1)
        # sample_j = sample_j[0]

        item_j = random.choice(j_choices)
        item_j_choices = items_to_all_responses[item_j]

        sample_j = random.choice(items_to_all_responses[item_j])

        new_cond_dict = copy.deepcopy(cond_dict)
        new_cond_dict[item_j] = sample_j
        
        _, dist_i = sampleTree(
            tree_i, 
            cond=new_cond_dict,
            DIST=True,
            sample='random')

        dissonance_i = dissonance(dist_i, response_i)
        sampled_dissonances.append(dissonance_i)

    sampled_dissonances = np.array(sampled_dissonances)

    alpha = np.sum(sampled_dissonances <= orig_dissonance) \
        / float(len(sampled_dissonances))

    return alpha


def beta_i(
    orig_dissonance, 
    cond_dict, 
    item_i, 
    trees, 
    items_to_all_responses, 
    num_samples=100):
    """Compute the beta parameter for item i.

    Beta measures the difficulty of reducing dissonance when
    response i is perturbed.

    Beta is calculated by:
        1. randomly sampling i
        2. recomputing the dissonance of i
        3. repeating step 1 to 2 n times
        4. calculating the proportion of times the recomputed dissonance is less than the original dissonance

    Args:
        orig_dissonance: original dissonance of item i
        cond_dict: dict that maps item names to actual responses
        item_i: name of item i
        trees: dictionary mapping each item to its corresponding tree
        items_to_all_responses: map item to all possible responses for that item
        num_samples: number of times of resampling

    Returns:
        proportion of sampled dissonance less than original dissonance
        nan if the response of item i is nan
    """

    tree_i = trees[item_i]

    sampled_dissonances = []

    _, dist_i = sampleTree(
        tree_i, 
        cond=cond_dict,
        DIST=True,
        sample='random')

    if len(items_to_all_responses[item_i]) == 0:
        return float('nan')

    for _ in range(num_samples):

        # sample_i = sampleTree(
        #     tree_i,
        #     cond={},
        #     sample='random',
        #     DIST=False,
        #     NUMSAMPLE=1)
        # sample_i = sample_i[0]

        sample_i = random.choice(items_to_all_responses[item_i])
        dissonance_i = dissonance(dist_i, sample_i)
        sampled_dissonances.append(dissonance_i)

    sampled_dissonances = np.array(sampled_dissonances)
    beta = np.sum(sampled_dissonances <= orig_dissonance) \
        / float(len(sampled_dissonances))

    return beta


def trivializationVector(
    orig_dissonance_vec, 
    cond_dict, 
    trees, 
    items_to_all_responses,
    parameter='alpha', 
    num_samples=100):
    """Compute the trivialization vector for all the items.

    The parameter may either be alpha or beta.

    Args:
        orig_dissonance_vec (1d np array): original dissonance vector
        cond_dict (dict): dict that maps item names to actual responses
        trees (dict): dictionary mapping each item to its corresponding tree
        items_to_all_responses (dict): map item to all possible responses for that item
        parameter (str): alpha or beta
        num_samples (int): number of times of resampling

    Returns:
        (1d np array) trivialization vector
    """

    if parameter == 'alpha':
        param_type = alpha_i
    elif parameter == 'beta':
        param_type = beta_i
    else:
        raise ValueError('Not a correct parameter type.')

    all_items = list(trees.keys())
    all_items.sort()

    param_vector = []

    for i, item_i in enumerate(all_items):
        orig_dissonance = orig_dissonance_vec[i]

        param = param_type(
            orig_dissonance, 
            cond_dict,
            item_i,
            trees,
            items_to_all_responses,
            num_samples=num_samples)

        param_vector.append(param)
        

    param_vector = np.array(param_vector)

    return param_vector


def itemsToAllResponses(df):
    """Map column name (items) to possible values (responses) that 
    the entries for that column can take.
    
    NOTE: exclude the NaNs

    Args:
        df (pd.DF): dataframe containing all the data

    Returns:
        dictionary
    """

    response_to_labels = {}
    for col_name in df.columns:
        response_to_labels['P' + col_name] = filter(
            lambda v: v==v, df[col_name].unique())

    return response_to_labels


def trivializationVectors(
    df, 
    dissonance_df, 
    tree_dir, 
    save_file=None,
    parameter='alpha', 
    num_samples=100):
    """Compute the trivialization vectors for each sample.

    Args:
        df: dataframe containing the data
        dissonance_df: dissonance vectors where rows
            are the dissonance vector for each instance
        tree_dir: directory that the trees were saved
        parameter: alpha or beta
        num_samples: number of times of resampling

    Returns:
        2d numpy array of size 
            (number of samples, number responses)
    """

    if dissonance_df.shape[0] != df.shape[0]:
        raise ValueError("Number of instances must be the same.")

    num_rows = dissonance_df.shape[0]
    col_names = dissonance_df.columns

    response_to_labels = itemsToAllResponses(df)

    trees = load_trees(tree_dir)

    all_vecs = np.empty(dissonance_df.shape)

    for row_index in range(num_rows):
        
        # cond_dict maps response names to actual values from the data
        cond_dict = {}
        for col_name in col_names:
            cond_dict[col_name] = df[col_name[1:]][row_index]

        dissonance_vector = dissonance_df.iloc[row_index]

        trivialization_vec = trivializationVector(
            dissonance_vector.values, 
            cond_dict, 
            trees, 
            response_to_labels,
            num_samples=num_samples,
            parameter=parameter)

        all_vecs[row_index] = trivialization_vec


    # save the dissonance vectors to file
    if save_file is not None:
        df = pd.DataFrame(
            data=all_vecs, 
            columns=col_names)
        df.to_csv(
            save_file, 
            index=None)

    return all_vecs


def probTrivialization(alpha_df, beta_df):
    """Calculate probability of trivialization or rationalization.

    Args:
        alpha_df: dataframe of alpha parameters
        beta_df: dataframe of beta parameters

    Returns:
        df of the same shape as alpha and beta
    """

    if alpha_df.shape != beta_df.shape:
        raise ValueError('Alpha and beta dataframes must be the same.')

    return alpha_df / (alpha_df + beta_df)

    
def belief_shift_simulation(
    items_to_all_responses, 
    items_to_response,
    items_to_dissonance, 
    trees,
    num_instances,
    response_out_filename,
    dissonance_out_filename,
    threshold,
    patience):
    """Simulate the belief shift for a single instance.

    The new vector is computed by:
        1. randomly pick some item i
        2. randomly pick some item j such that if we switch the response for j,
            the dissonance for i does decreases. NOTE: i can equal j
        3. replace response j with the new response
        4. repeat steps 1 to j n times

    If the simulation stops early, then the values for the rest of the time steps
    will not change.

    Args:
        items_to_all_responses (dict): maps items to all possible responses
        items_to_response (dict): maps items to the actual response for one instance
        items_to_dissonance (dict): maps items to dissonance for one instance
        trees (dict): maps item name to the corresponding tree
        num_instances (int): number of times to repeat the simulation
        response_out_filename (str): filename to save responses for the belief shift
        dissonance_out_filename (str): filename to save dissonances for the belief shift
        threshold (float): if the difference between two time steps is lower than 
            the threshold, then the simulation stops
        patience (int): time inteval to decide wether to stop simulations

    Returns:
        None
    """

    # the initial dissonances / responses count as 1 simulation
    num_instances -= 1

    round_ = lambda x: round(x, 4)

    all_items = copy.deepcopy(list(trees.keys()))

    # we do not want to choose an item i where the response is NaN 
    # because there will be no dissonance for that response
    for item, response in items_to_response.items():
        if isnan(response) and (item in all_items):
            all_items.remove(item)
    
    # the columns are the different responses (or dissonance) for different items
    # the rows are the timesteps
    response_belief_shift_df = pd.DataFrame(
        items_to_response,
        index=[0])
    dissonance_belief_shift_df = pd.DataFrame(
        items_to_dissonance,
        index=[0])

    total_dissonances = np.empty(patience + 1)
    total_dissonances.fill(np.nan)
    total_dissonances[-1] = dissonance_belief_shift_df.iloc[0].sum()

    # simulate belief shift `num_instances` number of times
    for i in range(num_instances):

        item_i = random.choice(all_items)
        response_i = items_to_response[item_i]
        tree_i = trees[item_i]
        dissonance_i = items_to_dissonance[item_i]

        # 5 is arbitrary. We just want it to be greater than 1, since
        # dissonance can not be greater than 1.
        new_dissonance = 5

        # list of items that will influence item i with item i itself
        j_choices = tree_i.significant_feature_weight_.keys()
        j_choices.append(item_i)
        random.shuffle(j_choices)

        for item_j in j_choices:

            all_response_j = items_to_all_responses[item_j]
            random.shuffle(all_response_j)

            # special case
            if item_j == item_i:

                _, dist_i = sampleTree(
                    tree_i, 
                    cond=items_to_response,
                    DIST=True,
                    sample='random')

                for response_j in all_response_j:
                    new_dissonance_i = dissonance(dist_i, response_j)
                    if round_(new_dissonance_i) < round_(dissonance_i):
                        new_response_j = response_j
                        new_dissonance = new_dissonance_i 
                        break
                        
                    else:
                        new_response_j = items_to_response[item_j]
                        new_dissonance = items_to_dissonance[item_i]

                break

            # iterate over possible responses for item j
            for response_j in all_response_j:
                new_items_to_response = copy.deepcopy(items_to_response)
                new_items_to_response[item_j] = response_j
                
                _, dist_i = sampleTree(
                    tree_i, 
                    cond=new_items_to_response,
                    DIST=True,
                    sample='random')

                new_dissonance_i = dissonance(dist_i, response_i)
                new_dissonance = new_dissonance_i

                # stop loop if we find a lower dissonance for item i
                if round_(new_dissonance_i) < round_(dissonance_i):
                    new_response_j = response_j
                    break

                # if we never have that the new dissonance is lower than the 
                # original dissonance, then the response for item j remains the same
                else:
                    new_response_j = items_to_response[item_j]
                    new_dissonance = items_to_dissonance[item_i]

            # stop the outer loop when the inner loop also has been stopped
            if round_(new_dissonance_i) < round_(dissonance_i):
                break
    
        items_to_response[item_j] = new_response_j
        items_to_dissonance[item_i] = new_dissonance

        response_belief_shift_df = response_belief_shift_df.append(
            items_to_response, 
            ignore_index=True)

        dissonance_belief_shift_df = dissonance_belief_shift_df.append(
            items_to_dissonance, 
            ignore_index=True)

        new_total_dissonance = dissonance_belief_shift_df.iloc[-1].sum()
        total_dissonances = np.roll(total_dissonances, -1)
        total_dissonances[-1] = new_total_dissonance

        # i.e., if the patience has ran out
        if total_dissonances[0] - total_dissonances[-1] < threshold:
            break

    response_belief_shift_df.sort_index(axis=1, inplace=True)
    dissonance_belief_shift_df.sort_index(axis=1, inplace=True)
    print(i)
    response_belief_shift_df.to_csv(
        response_out_filename, 
        index=None)

    dissonance_belief_shift_df.to_csv(
        dissonance_out_filename, 
        index=None)


def bs_sim_multiprocessing(args):
    """This is the same as belief_shift_simulation, but this 
    is used for multiprocessing.
    """

    belief_shift_simulation(
        items_to_all_responses=args[0], 
        items_to_response=args[1],
        items_to_dissonance=args[2], 
        trees=args[3],
        num_instances=args[4],
        response_out_filename=args[5], 
        dissonance_out_filename=args[6],
        threshold=args[7],
        patience=args[8])


def belief_shift_simulations(
    df, 
    dissonance_df,
    tree_dir, 
    num_instances,
    belief_shift_dir,
    threshold=0.05,
    patience=10,
    numCPUs=None):
    """Simulate the belief shift for all the data.

    Args:
        df (pd.DF): dataframe containing the all the data
        dissonance_df (pd.DF): dissonance vectors where rows
            are the dissonance vector for each instance
        tree_dir (str): directory where the trees were saved
        num_instances (int): number of times to repeat the simulation
            for each sample instance
        belief_shift_dir (str): directory to save belief shift results
        threshold (float): if the difference between two time steps is lower than 
            the threshold, then the simulation stops
        patience (int): time inteval to decide wether to stop simulations
        numCPUs (int): number of CPUs to use for the simulation

    Returns:
        (pd.DF) dataframe for belief shifted responses
        (pd.DF) dataframe for belief shifted dissonances
    """

    if df.shape[0] != dissonance_df.shape[0]:
        raise ValueError('df and dissonance_df must have the same number of instances.')
    
    if numCPUs is None:
        numCPUs = multiprocessing.cpu_count()

    items_to_all_responses = itemsToAllResponses(df)
    trees = load_trees(tree_dir)

    num_samples = df.shape[0]

    new_col_names = ['P' + col_name for col_name in df.columns]
    df.columns = new_col_names

    arguments_set = []
    for i in range(num_samples):
        arguments_set.append([
            items_to_all_responses,
            df[i:i+1].to_dict('records')[0],
            dissonance_df[i:i+1].to_dict('records')[0],
            trees,
            num_instances,
            belief_shift_dir + '/response{}.csv'.format(i),
            belief_shift_dir + '/dissonance{}.csv'.format(i),
            threshold,
            patience])

    pool = multiprocessing.Pool(numCPUs)
    pool.map(bs_sim_multiprocessing, arguments_set)


