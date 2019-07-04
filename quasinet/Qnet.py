import glob
from graphviz import Digraph
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import operator
import os
import pandas as pd
import pickle
from quasinet import mlx
import re
from sklearn.model_selection import train_test_split


def fit_sequences(sequence_file,trainfile,testfile, test_ratio=0.5):
	'''
	Takes a file of sequences and breaks it up into train and test csvs
	appropriate for qnets.
	Inputs-
		sequence_file (str)- path to file of sequences)
		trainfile (str)- path to write train csv to.
		testfile (str)- path to write test csv to.
	'''
	
	with open(sequence_file,'r') as fh:
		contents = fh.readlines()
		contents = [sequence.strip('\n') for sequence in contents]
	
	df = pd.DataFrame(list(seq) for seq in contents)
	train_df, test_df = train_test_split(df,test_size=test_ratio)
	
	train_df.to_csv(trainfile,index=None)
	test_df.to_csv(testfile,index=None)
	
	return train_df, test_df


def singleTree(args):
	'''
	Given a selected response position, generates the conditional
	inference tree for that position as a pickle file. Uses a train
	and test csv to do so. Saves the tree as a picle file.
	Inputs-
		response (int)- an integer indicating the index or position of the
			variable which is being regressed against the other variables.
		trainfile(str)- the csv file containing the training sequences.
		testfile(str)- the csv file containing the test sequences.
		tree_dir(str)- directory to store tree.
	'''
	response = args[0]
	trainfile = args[1]
	testfile = args[2]
	tree_dir = args[3]
	VERBOSE = args[4]
	if VERBOSE:
		print "Generating tree for response {}".format(response)

	R = 'P' + str(response)
	datatrain = mlx.setdataframe(trainfile)
	datatest = mlx.setdataframe(testfile)
	
	CT,Pr,ACC,CF,Prx,ACCx,CFx,TR = mlx.Xctree(RESPONSE__=R,
									datatrain__=datatrain,
									datatest__=datatest,
									VERBOSE=False,
									TREE_EXPORT=False)

	pickle_file = tree_dir + R + '.pkl'
	dot_file = tree_dir + R + '.dot'

	with open(pickle_file,'w') as fh:
		pickle.dump(TR,fh)

	mlx.tree_export(TR, outfilename=dot_file, EXEC=False)


def makeQNetwork(response_set,trainfile, testfile, tree_dir='tree/',VERBOSE=False):
	'''
	Given a set of responses, will generate a QNet with a tree representing
	each response variable. 
	Inputs-
		response (int)- an integer indicating the index or position of the
			variable which is being regressed against the other variables.
		trainfile(str)- the csv file containing the training sequences.
		testfile(str)- the csv file containing the test sequences.
		tree_dir(str)- directory to store trees.
	'''
	if not os.path.isdir(tree_dir):
		os.mkdir(tree_dir)

	arguments_set = [[R, trainfile, testfile, tree_dir,VERBOSE] for R in response_set]

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(singleTree,arguments_set)


def processEdgeUpdate(edges_):
	SOURCES_=[i[0] for  i in edges_.keys() if edges_[i]>0.0]
	PROCESSED_=list(set([i[1] for  i in edges_.keys()]))
	return SOURCES_,PROCESSED_


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


def getDot(edges,DOTFILE='out.dot',EDGEFILE=None):
	dot = Digraph()
	for key,values in edges.iteritems():
			if key[0] is not "":
				dot.edge(key[0],key[1])
	f1=open(DOTFILE,'w+')
	f1.write(dot.source)
	f1.close()

	if EDGEFILE is not None:
		df1=pd.DataFrame.from_dict(edges,orient='index')
		df1.columns=['imp']
		df1=df1[df1.imp>0.0]
		df1.to_csv(EDGEFILE,header=None,sep=",")

	return


def connectQnet(RS,FEATURE_IMP_THRESHOLD,DOTFILE,EDGEFILE,tree_dir='tree/',DEBUG=False):
	'''
	For the purpose of generating qNetwork. We go through each node, and 
	examine if each connection between nodes satistifes a given 
	feature importance threshold. In the end, dat and dot files will be
	produced, which will represent qNetworkds, which we will draw.
	Input
		RS(list of string): List of strings denoting the responses.
			Ex: '0','1','2',....
		FEATURE_IMP_THRESHOLD(float): a float denoting the threshold used
			to evaluate significant connections. Ex: 0.75
		DOTFILE (str): file to write a dot representation of qnetwork.
		EDGEFILE(str): file to write edges of qnetwork to.
		tree_dir (str): directory to where the trees are stored.

	'''
	RS = ['P' + str(R) for R in RS]
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	edges = {}
	SOURCES=[]
	PROCESSED=[]
	DIFF=[]
	while RS is not None:
		args = [[R, FEATURE_IMP_THRESHOLD,tree_dir] for R in RS]
		edges__ = pool.map(apply_threshold_to_tree,args)

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

		getDot(edges,DOTFILE=DOTFILE,EDGEFILE=EDGEFILE)

	pool.close()
	pool.join()


def apply_threshold_to_tree(args):
	'''
	Given a response and a feature importance threshold, loads in 
	the conditional inference tree representing that response. Then
	examines and determines which connections meet the threshold
	requirement.
	Inputs-
		R(str)- string representing the response. Ex: 'P10'
		FEATURE_IMP_THRESHOLD(float): a float denoting the threshold used
			to evaluate significant connections. Ex: 0.75
		tree_dir (str): directory to where the trees are stored.
	'''
	R = str(args[0])
	FEATURE_IMP_THRESHOLD = args[1]
	tree_dir = args[2]

	edges_={}
	glob_str = tree_dir + '*' + R + '.pkl'
	found_files = glob.glob(glob_str)
	if len(found_files) == 0:
		edges_ = {('',R):0.0}
		return edges_
	file = found_files[0]

	with open(file, 'r') as fh:
		TR = pickle.load(fh)

	sorted_feature_imp\
	= sorted(TR.significant_feature_weight_.items(),
								key=operator.itemgetter(1))
	edges_ = {(i[0],R):i[1] for i in sorted_feature_imp
		if i[1] > FEATURE_IMP_THRESHOLD  }
	if not edges_:
		edges_={('',R):0.0}
	return edges_


def draw_Qnet(edgefile,out_name='qnet.png',dim=(50,50)):
	'''
	This function is used to draw a representation of the qNet
	from the edgefile produced by the connectQnet function.
	Inputs-
		edgefile (str): path to file representing the edges of Qnet
		out_name (str): name of file to save the drawn Qnet.
		dim (tuple of ints): dimensions of the plot.
	'''
	dotpattern = r'P([0-9]+) -> P([0-9]+)'
	edges = set()
	nodes = set()

	with open(edgefile, 'r') as fh:
		text = fh.read()
		for match in re.finditer(dotpattern, text):
			x, y = int(match.group(1)), int(match.group(2))
			edges.add((x, y))
			nodes.add(x)
			nodes.add(y)

	fig = plt.figure(figsize=dim)
	fig.tight_layout()
	graph = nx.DiGraph()
	graph.add_nodes_from(nodes)
	graph.add_edges_from(edges)
	pos = nx.nx_agraph.graphviz_layout(graph)
	nx.draw(
		graph,
		pos=pos,
		with_labels = True,
		node_size = 1000,
		node_color = 'white',
	)
	ax = plt.gca() # to get the current axis
	ax.collections[0].set_edgecolor("#000000")

	print("Saving to {}".format(out_name))
	plt.savefig(
		out_name,
		bbox_inches = 'tight',
	)
	plt.close()
