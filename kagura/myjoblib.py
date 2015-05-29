"""
joblib wrapper: put files in a directory
"""

from sklearn.externals import joblib

def load(filename):
    filepath = os.path.join(filename, 'index')
    if os.path.isfile(filepath):
        return joblib.load(filepath)
    raise RuntimeError('No file', filepath)


def dump(value, filename):
    import os
    os.makedirs(filename)
    filepath = os.path.join(filename, 'index')
    joblib.dump(value, filepath)

