"""
joblib wrapper: put files in a directory
"""

from sklearn.externals import joblib
import os

def load(filename):
    filepath = os.path.join(filename, 'index')
    if os.path.isfile(filepath):
        return joblib.load(filepath)
    raise RuntimeError('No file', filepath)


def dump(value, filename, force_overwrite=False):
    import os
    if os.path.exists(filename):
        if not force_overwrite:
            # user specified an existing filename, avoid overwriting
            dump(value, filename + "_NEW")
            return
    else:
        os.makedirs(filename)
    filepath = os.path.join(filename, 'index')
    joblib.dump(value, filepath)

