"""
joblib wrapper: put files in a directory.

aliasd as kagura.load and kagura.dump
"""

from sklearn.externals import joblib
import os

def load(name):
    "load from given name"
    filepath = os.path.join(name, 'index')
    if os.path.isfile(filepath):
        return joblib.load(filepath)
    raise RuntimeError('No file', filepath)


def dump(value, name, force_overwrite=False):
    """dump value as given name.
    """
    import os
    if os.path.exists(name):
        if not force_overwrite:
            # user specified an existing name, avoid overwriting
            dump(value, name + "_NEW")
            return
    else:
        os.makedirs(name)
    filepath = os.path.join(name, 'index')
    joblib.dump(value, filepath)


def make_if_not_exist(builder, name, force_regenerate=False):
    if not force_regenerate and os.path.exists(name):
        return load(name)
    value = builder()
    dump(value, name)
    return value
