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
            n_estimators=int(args.param), criterion='entropy')
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

    if extention:
        model = extention()
        if model:
            return model

    raise NotImplementedError


def add_default_to_get_model(f):
    return lambda args: get_model(args, f)
