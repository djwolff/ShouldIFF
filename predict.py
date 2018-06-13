import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from io import StringIO
import pydot
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from plotting import plot_learning_curve

## Get data
dataset = pandas.read_csv('data/seasons/season8.csv')

# To determine how many samples we want.
n = 100000

# X = data and y = target
array = dataset.values
X = array[:n,1:]  	# The data and attributes
y = array[:n,0]		# The targets

## Initialization of our data
validation_size = 0.2
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


## Debugging issues
# print("Classifier: %s" % dataset.columns.values[0])
# print("Attributes: %s" % dataset.columns.values[1:])

## From the models we looked at, we choose DecisionTreeClassifier (CART).
estimator = DecisionTreeClassifier(random_state=0)
estimator.fit(X_train, Y_train)

# The decision estimator has an attribute called tree_  which stores the entire
# tree structure and allows access to low level attributes. The binary tree
# tree_ is represented as a number of parallel arrays. The i-th element of each
# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
# Some of the arrays only apply to either leaves or split nodes, resp. In this
# case the values of nodes of the other type are arbitrary!
#
# Among those arrays, we have:
#   - left_child, id of the left child of the node
#   - right_child, id of the right child of the node
#   - feature, feature used for splitting the node
#   - threshold, threshold value at the node
#

# Using those arrays, we can parse the tree structure:

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = numpy.zeros(shape=n_nodes, dtype=numpy.int64)
is_leaves = numpy.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

#Create Dictionary of Rows -> Attribute Title
attribute_dict = {
"0":"teamid",
"1":"totalwards",
"2":"hasAhri",
"3":"hasAkali",
"4":"hasAlistar",
"5":"hasAmumu",
"6":"hasAnivia",
"7":"hasAnnie",
"8":"hasAshe",
"9":"hasAurelionSol",
"10":"hasAzir",
"11":"hasBard",
"12":"hasBlitzcrank",
"13":"hasBrand",
"14":"hasBraum",
"15":"hasCaitlyn",
"16":"hasCamille",
"17":"hasCassiopeia",
"18":"hasChoGath",
"19":"hasCorki",
"20":"hasDarius",
"21":"hasDiana",
"22":"hasDraven",
"23":"hasDrMundo",
"24":"hasEkko",
"25":"hasElise",
"26":"hasEvelynn",
"27":"hasEzreal",
"28":"hasFiddlesticks",
"29":"hasFiora",
"30":"hasFizz",
"31":"hasGalio",
"32":"hasGangplank",
"33":"hasGaren",
"34":"hasGnar",
"35":"hasGragas",
"36":"hasGraves",
"37":"hasHecarim",
"38":"hasHeimerdinger",
"39":"hasIllaoi",
"40":"hasIrelia",
"41":"hasIvern",
"42":"hasJanna",
"43":"hasJarvanIV",
"44":"hasJax",
"45":"hasJayce",
"46":"hasJhin",
"47":"hasJinx",
"48":"hasKalista",
"49":"hasKarma",
"50":"hasKarthus",
"51":"hasKassadin",
"52":"hasKatarina",
"53":"hasKayle",
"54":"hasKennen",
"55":"hasKhaZix",
"56":"hasKindred",
"57":"hasKled",
"58":"hasKogMaw",
"59":"hasLeBlanc",
"60":"hasLeeSin",
"61":"hasLeona",
"62":"hasLissandra",
"63":"hasLucian",
"64":"hasLulu",
"65":"hasLux",
"66":"hasMalphite",
"67":"hasMalzahar",
"68":"hasMaokai",
"69":"hasMasterYi",
"70":"hasMissFortune",
"71":"hasMordekaiser",
"72":"hasMorgana",
"73":"hasNami",
"74":"hasNasus",
"75":"hasNautilus",
"76":"hasNidalee",
"77":"hasNocturne",
"78":"hasNunu",
"79":"hasOlaf",
"80":"hasOrianna",
"81":"hasPantheon",
"82":"hasPoppy",
"83":"hasQuinn",
"84":"hasRakan",
"85":"hasRammus",
"86":"hasRekSai",
"87":"hasRenekton",
"88":"hasRengar",
"89":"hasRiven",
"90":"hasRumble",
"91":"hasRyze",
"92":"hasSejuani",
"93":"hasShaco",
"94":"hasShen",
"95":"hasShyvana",
"96":"hasSinged",
"97":"hasSion",
"98":"hasSivir",
"99":"hasSkarner",
"100":"hasSona",
"101":"hasSoraka",
"102":"hasSwain",
"103":"hasSyndra",
"104":"hasTahmKench",
"105":"hasTaliyah",
"106":"hasTalon",
"107":"hasTaric",
"108":"hasTeemo",
"109":"hasThresh",
"110":"hasTristana",
"111":"hasTrundle",
"112":"hasTryndamere",
"113":"hasTwistedFate",
"114":"hasTwitch",
"115":"hasUdyr",
"116":"hasUrgot",
"117":"hasVarus",
"118":"hasVayne",
"119":"hasVeigar",
"120":"hasVelKoz",
"121":"hasVi",
"122":"hasViktor",
"123":"hasVladimir",
"124":"hasVolibear",
"125":"hasWarwick",
"126":"hasWukong",
"127":"hasXayah",
"128":"hasXerath",
"129":"hasXinZhao",
"130":"hasYasuo",
"131":"hasYorick",
"132":"hasZac",
"133":"hasZed",
"134":"hasZiggs",
"135":"hasZilean",
"136":"hasZyra",
"137":"first_blood_team",
"138":"firsttower",
"139":"firstinhib",
"140":"firstbaron",
"141":"firstdragon",
"142":"firstharry"
}


# print("The binary tree structure has %s nodes and has "
#       "the following tree structure:"
#       % n_nodes)
# for i in range(n_nodes):
#     if is_leaves[i]:
#         print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
#     else:
#         print("%snode=%s test node: go to node %s if %s <= %s else to "
#               "node %s."
#               % (node_depth[i] * "\t",
#                  i,
#                  children_left[i],
#                  attribute_dict[str(feature[i])],
#                  threshold[i],
#                  children_right[i],
#                  ))
# print()

# First let's retrieve the decision path of each sample. The decision_path
# method allows to retrieve the node indicator functions. A non zero element of
# indicator matrix at the position (i, j) indicates that the sample i goes
# through the node j.

node_indicator = estimator.decision_path(X_validation)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = estimator.apply(X_validation)

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample.

sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

print('Rules used to predict sample %s: ' % sample_id)
for node_id in node_index:
    if leave_id[sample_id] != node_id:
        continue

    if (X_validation[sample_id, feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"

    print("decision id node %s : (X_validation[%s, %s] (= %s) %s %s)"
          % (node_id,
             sample_id,
             feature[node_id],
             X_validation[sample_id, feature[node_id]],
             threshold_sign,
             threshold[node_id]))

# For a group of samples, we have the following common node.
sample_ids = [0, 1]
common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                len(sample_ids))

common_node_id = numpy.arange(n_nodes)[common_nodes]

print("\nThe following samples %s share the node %s in the tree"
      % (sample_ids, common_node_id))
print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))


######## Create a graph!!
import graphviz
dot_data = StringIO()
tree.export_graphviz(estimator, out_file=dot_data, feature_names=dataset.columns.values[1:], class_names=["0", "1"], special_characters=True, max_depth=4)
graph = graphviz.Source(dot_data.getvalue())
graph.render("tree", view=True)


######## Now to predict!!!

ourGame = numpy.array([dataset.values[len(dataset.values)-2,1:]]) # Choose a datavalue that we haven't seen.
result = estimator.predict(ourGame)

if result[0] == 0:
	print("If the game is prolonged, your team will lose. It is wise to surrender now.")
if result[0] == 1:
	print("Don't give up! Your team is likely to win this!")
