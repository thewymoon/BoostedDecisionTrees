



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



