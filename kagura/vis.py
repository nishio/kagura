"""
visualize
"""
from sklearn.tree import export_graphviz
import subprocess
def viz_tree(X, Y, labels=None):
    m = DecisionTreeClassifier(max_depth=5)
    m.fit(X, Y)
    export_graphviz(m, 'tree.dot', feature_names=labels)
    subprocess.check_call('dot -Tpng tree.dot -o tree.png'.split())
    return m
