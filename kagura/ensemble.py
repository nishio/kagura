# -*- encoding: utf-8 -*-
"""
アンサンブル。再利用するのでとりあえずここに置いたけど
他の関数への依存がヘビーなのでこのままでは動かない。
整理が必要。

どっちみちこのLRのアンサンブラは
今後NNとかGBDTを使えるように修正される予定。
"""


def get_blend(filename):
    blends = [load(name) for name in filename.split(",")]
    blend_train = np.hstack(v[0] for v in blends)
    blend_test = np.hstack(v[1] for v in blends)
    assert len(blend_train) == NUM_TRAIN
    assert len(blend_test) == NUM_TEST
    return blend_train, blend_test


def ensemble():
    ys = load('ys')
    logging.info("ensenble %s", args.ensemble)
    blend_train, blend_test = get_blend(args.ensemble)

    # dump blended data
    name = "blend_%s" % args.name
    logging.info("writing blended data to %s", name)
    dump((blend_train, blend_test), name)

    logging.info("training final LR")
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1000)
    model.fit(blend_train, ys)
    logging.info("trained final LR")
    ys_sub = model.predict(blend_test)
    write_submit(ys_sub)
