"""
USAGE without extention:
get_model(args)

USAGE with extention:
@add_default_to_get_model
def get_model(args):
    # your own extention here
"""

def get_model(args, extention=None):
    """
    get args, return new model instance
    """
    m = args.model
    if m == "Dummy":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier()
    if m == "LR":
        from sklearn.linear_model import LogisticRegression
        if not args.param:
            args.param = 1
        return LogisticRegression(C=float(args.param))
    if m == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(args.param), criterion='entropy')
    if m == "RF_gini":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(args.param), criterion='gini')
    if m == "NB":
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB()
    if m == "ExT":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=int(args.param), criterion='entropy')
    if m == "ExT_gini":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=int(args.param), criterion='gini')
    if m == "GBC" or m == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=int(args.param))
    if m == "LRL1":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(penalty='l1', C=0.01)
    if m == "SVC":
        from sklearn.svm import SVC
        return SVC(kernel=args.param, probability=True)
    if m == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier()
    if m == "XGB":
        from .xgbwrapper import XGBWrapper
        return XGBWrapper()
    if m == "XGBBin":
        from .xgbwrapper import XGBBinary
        if not args.param: args.param = '0.3'
        return XGBBinary(eta=float(args.param))

    if m == "NN":
        from .mylasagne import LasagneWrapper
        return LasagneWrapper()
    if m == "PCA_XGBBin":
        from .xgbwrapper import XGBBinary
        if not args.param: args.param = '0.3'
        return PCAPreprocess(XGBBinary(eta=float(args.param)))
    if m == "SS_KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return SubsamplePreprocess(KNeighborsClassifier())
    if extention:
        model = extention()
        if model:
            return model

    raise NotImplementedError


def add_default_to_get_model(f):
    return lambda args: get_model(args, f)


class PCAPreprocess(object):
    def __init__(self, model):
        self.model = model
    def fit(self, xs, ys):
        from sklearn.decomposition import PCA
        self.pca = m = PCA(n_components=len(xs[0]), whiten=True)
        xs_pca = m.fit_transform(xs)
        self.model.fit(xs_pca, ys)
        return self
    def predict_proba(self, xs):
        xs_pca = self.pca.transform(xs)
        return self.predict_proba(xs_pca)


class SubsamplePreprocess(object):
    def __init__(self, model, p=0.1):
        self.model = model
        self.p = p
    def fit(self, xs, ys):
        r = np.random.random(len(xs))
        sample = (r < self.p)
        xs = xs[sample]
        ys = ys[sample]
        self.model.fit(xs, ys)
        return self
    def predict_proba(self, xs):
        return self.model.predict_proba(xs)
