from quasinet import Qnet
import subprocess

responses = list(range(0,400))

#Step 1:
#Qnet.fit_sequences('sequences.txt','train.csv','test.csv', test_ratio=0.2)

#Step 2:
#Qnet.makeQNetwork(responses, 'train.csv','test.csv',tree_dir='tree/', VERBOSE=True)

#Step 3:
#Qnet.connectQnet(responses, 0.50, '50network.dot','50network.dat',tree_dir='tree/',DEBUG=True)
#Qnet.draw_Qnet('50network.dot',out_name = '50net.png')

#Step 4:
#Qnet.connectQnet(responses, 0.75, '75network.dot','75network.dat',tree_dir='tree/',DEBUG=True)
#Qnet.draw_Qnet('75network.dot',out_name = '75net.png')

#Step 5:
#subprocess.Popen(["dot", '-Tpng', 'tree/P25.dot', '-o', 'decision_tree25.png'])
