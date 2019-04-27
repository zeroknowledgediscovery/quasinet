import string
import random
import numpy as np
import pandas as pd
import subprocess
import os
import sys
import decimal

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

def get_terminal_distinction_coeff(terminal_class_vector,
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

def feature_importance(significant_dec_path,
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

def visTree(MODEL,PR,PLOT=True,VERBOSE=False,
            ACC=None,ACCx=None,RESP_=None,PROB_MIN=0.1):
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
            tmpfilename_=id_generator(8)
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

#        if PLOT:
#            xplot(CLASSES,100*CFRQ[count])
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
    if unicode(str_).isdigit():
        str_='P'+str_

    return str_

def setdataframe(file1,outname="",
                 delete_=[],include_=[],
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
    datatrain = pd.DataFrame(X_train,columns=D1.columns).dropna('columns')

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
    return ''.join(random.choice(chars) for _ in range(size))

#------------------------------------------

def getresponseframe(DATA,MODEL,RESPONSE_,olddata=False):
    tmpfilename=id_generator(8)
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
    tmpfilename=id_generator(8)
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
    tmpfilename=id_generator(8)
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
nodeapply=robjects.r('nodeapply')
assimpleparty=robjects.r('as.simpleparty')
infonode=robjects.r('info_node')
sapply_=robjects.r('sapply')
#------------------------------

def sapply(node,arg2="[[",criteria="error"):
    return sapply_(node,arg2,criteria)


#------------------------------------------------------------

def Xctree(RESPONSE__,
           datatrain__,
           datatest__=None,
           VERBOSE=False,
           TREE_EXPORT=True):

    Prx__=None
    ACCx__=None
    CFx__=None

    fmla__ = Formula(RESPONSE__+' ~ .')
    CT__ = ctree(fmla__,
                data=datatrain__)
    Pr__,ACC__,CF__= getresponseframe(datatrain__,CT__,
                                        RESPONSE__,olddata=True)
    if datatest__ is not None:
        Prx__,ACCx__,CFx__= getresponseframe(datatest__,CT__,
                                             RESPONSE__)
    TR__= visTree(CT__,Pr__,
                    PLOT=False,
                    VERBOSE=VERBOSE,ACC=ACC__,ACCx=ACCx__,RESP_=RESPONSE__)
    if TR__ is not None:
        if TREE_EXPORT:
            tree_export(TR__,TYPE='polyline',EXEC=True)
    return CT__,Pr__,ACC__,CF__,Prx__,ACCx__,CFx__,TR__

#------------------------------------------------------------


def tree_export(TR,outfilename='out.dot',
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




#------------------------------------------------------------
#---------------------------------------------------------------
def randomForestX(RESPONSE__,
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
    
def sampleTree(tree,cond={},sample='mle',DIST=False,NUMSAMPLE=10):
    '''
        draw sample from decision tree
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
        
        2. sample: 'mle'|'random' 
                   if 'mle' then return the value with maximum probability.
                   if 'random' then makes random choice NUMSAMPLE times 
                   and returns the result.
        
        3. DIST: TRUE|FALSE
                 if TRUE returns the distribution from the tree 
                 after applying the constraints
    '''
    dist_=getMergedDistribution(tree,cond=cond)
    if sample is 'mle':
        sample=max(dist_.iteritems(), key=operator.itemgetter(1))[0]
    if sample is 'random':
        probs = dist_.values()
        keys =  dist_.keys()

        sample = np.random.choice(keys,NUMSAMPLE, replace=True, p=probs)
    if DIST:
        return sample,dist_
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
