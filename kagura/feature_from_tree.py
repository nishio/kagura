"""
extract features from decision tree

"""
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier

def pretty_features(features):
    return " and ".join("{} {} {:.2f}".format(*x) for x in features)

def generate_code(features):
    cond = " and ".join(
        "row[{}] {} {:.2f}".format(f, op, th)
        for (t, op, th, f) in features)
    return "  np.array([[1] if {} else [0] for row in X]),".format(cond)

def print_features(X, Y, labels):
    def get_label(i):
        if i < len(labels):
            return labels[i]
        return "X[%d]" % i

    m = DecisionTreeClassifier(max_depth=5)
    m.fit(X, Y)

    buf = []
    def visit(node_id, features=[]):
        left = m.tree_.children_left[node_id]
        if left != -1:
            f = m.tree_.feature[node_id]
            th = m.tree_.threshold[node_id]
            features.append((get_label(f), '<=', th, f))
            lvalue = visit(left)
            pf = pretty_features(features)
            code = generate_code(features)
            features.pop(-1)

            right = m.tree_.children_right[node_id]
            features.append((get_label(f), '>', th, f))
            rvalue = visit(right)
            features.pop(-1)

            a, pvalue, b, c = chi2_contingency([lvalue, rvalue])
            buf.append((pvalue, pf, code, lvalue, rvalue))
            return lvalue + rvalue
        else:
            return m.tree_.value[node_id]

    visit(0)
    buf.sort()
    print "X = np.hstack([X, "
    for (pvalue, pf, code, lvalue, rvalue) in buf:
        if pvalue > 0.05: break
        print "# {} left: {}, right: {}, p-value: {:.3f}\n{}".format(
            pf, lvalue, rvalue, pvalue, code)
    print "])"
