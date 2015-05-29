"""
unmature
"""

def initial_parse_given_csv():
    import pandas as pd
    train = pd.read_csv('sampleSubmission.csv')
    id = train.Id.values
    assert len(id) == NUM_TEST
    dump(id, 'id')

    train = pd.read_csv('train.csv')
    xs = train.drop('Id', axis=1)
    xs = xs.drop('Cover_Type', axis=1).values
    ys = train.Cover_Type.values
    assert len(xs) == NUM_TRAIN
    assert len(ys) == NUM_TRAIN
    assert len(set(ys)) == NUM_CLASSES
    dump(xs, 'xs')
    dump(ys, 'ys')

    test = pd.read_csv('test.csv')
    xs_sub = test.drop('Id', axis=1).values
    assert len(xs_sub) == NUM_TEST
    dump(xs_sub, 'xs_sub')
