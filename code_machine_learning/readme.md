 # This is the code used for the paper :  
"Restore structure evolution trajectory of networked complex systems. "

If you use this repository, please cite the manuscript.


In order to run the following code please in your terminal (preferably in a conda environment):

$pip install -r requirements.txt


# Main Code:
 "_ShowModelAuc.py " provides the accuracy of the machine learning algorithm to predict generation order of edge pairs，including different methods such as best feature, collection feature, deepwalk, node2vec, line, struct2vec, sdne, ensemble；
 "_ShowBaseModelAuc_.py " provides the accuracy of CPNN model with 11 single edge index;
 "_ShowEdgeSort.py " provides edge restoration results on real-world networks，including a comparison of Borda Count and Simulated Annealing；
 "_ShowDiffEdgeRatioModleAuc_.py " provides the influence of different training proportion on the model’s accuracy;
 "_ShowTheoreticalResult_.py " provides theoretical relation between the edge pairs order prediction accuracy and the error of the reconstructed edge sequence;
 "_CrossNetsJudge_direct.py " provides the code of transferring learning, using the ensemble model trained by one Synthetic network to judge the generation order of edge pairs in another Synthetic network directly; 
 "_CrossNetsJudge_linear.py " is similar to the code _CrossNetsJudge_direct.py, but it uses vector transformation of nodes instead of direct transformation in transfer learning;


# Basic Code:
 "getDict.py " : input net name, return all times of the networks ;
 "getNet.py " : input net name, return adjacency matrix, edges, nodes and network structure;
 "getEdgeTime.py " : input net name, return edge time, or input two edges, return the one which is older, or input net name, rank the times;
 "getEdgePair.py " : input net name, return the construction of training set and test set in our method;



# 11 basic edge features and the best feature
 "getFeatureJudge.py " : input net name and adjacency matrix, return the value of 11 basic features:
	Edge betweenness, Edge degree, Common neighbor, Edge clustering coefficient, 
	Edge strength, Resource allocation index, Adamic-Adar index,
 	Preferential attachment index,Local path index, Edge PageRank, Edge k-shell;
   	or input two edges, feature names and their feature values, return 0 if the feature of the edge1 bigger than edge2, if equal, return -1, otherwise, return 1;
 "SingleJudge_.py" :  input net name, train edges, test edges, return the test accuary of 11 basic features 
	(no cpnn model train, just according to the feature accuary to predict which edge is older or newer.)
"_BestSingleJudge_.py" : input net name, train edgeparis and label , return the best feature of 11 basic features.



# 7 edge repersentations and model
 "getEdgeEmbedding.py " : input net name, return node embedding vectors, edge embedding vectors of 7 edge repersentation:
	best feature, collection feature, deepwalk, node2vec, line, struct2vec, sdne；
"_BaseModelJudge_.py" : input any judge methods, including 7 edge representations and 11 basic features methods, return its' corresponding model.
 "_PairStructureNNjudge_.py": construct CPNN model in our paper 
 "_EnsembleJudge_.py": input judge methods,return model and best wights to construct ensemble model.



### The files in the fold of deepwalk, node2vec, sdne, struct2vec and line are downloaded from https://github.com/shenweichen/GraphEmbedding.


### If you have any questions, you can contact the author via <wangjy293@gmail.com>.