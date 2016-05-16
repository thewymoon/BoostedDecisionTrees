import numpy as np
import pandas as pd



def add_information_from_tree(tree, matrix):
    zipped = zip(tree.children_left, tree.children_right, tree.feature, tree.threshold)
    for i in range(len(zipped)):
        if zipped[i][0] < 0:
            pass
        else:
            parent = zipped[i][2]
            left_child = zipped[zipped[i][0]][2]
            right_child = zipped[zipped[i][1]][2]
            
            matrix[parent][left_child] += 1
            matrix[parent][right_child] += 1
    
    return matrix

def get_directed_matrix(bdt):
    connections = np.zeros((len(branch_names), len(branch_names)))
    for tree in bdt.estimators_:
        connections = add_information_from_tree(tree.tree_, connections)
    return connections
def get_undirected_matrix(bdt):
    connections = get_directed_matrix(bdt)
    for i in range(len(connections)):
        for j in range(i):
            connections[i][j] += connections[j][i]
            connections[j][i] = 0
    return connections



