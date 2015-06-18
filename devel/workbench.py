#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Binary Classification
"""
import numpy as np
from kagura.getlogger import logging
from kagura.safetynet import safetynet, hole
from kagura.utils import HumaneElapse
from kagura.getarg import get_args, get_default_arg_parser, make_better_name
from kagura import processqueue
from kagura.getmodel import get_model
from kagura.myjoblib import load, dump


from sklearn.cross_validation import StratifiedKFold
NUM_FOLDS = 5
IS_BINARY_CLASSIFICATION = True

def get_train_data():
    # 今回は特徴量の変換などはしないのでシンプル
    return load('xs')


def get_test_data():
    return load('xs_sub')


def make_white_data():
    from sklearn.preprocessing import StandardScaler, LabelBinarizer

    train = load('train_pd')
    test = load('test_pd')
    spray_train = load('spray_train')
    spray_test = load('spray_test')

    lb_species = LabelBinarizer().fit(list(train.Species) + list(test.Species))
    species_train = lb_species.transform(train.Species)
    species_test = lb_species.transform(test.Species)

    lb_street = LabelBinarizer().fit(list(train.Street) + list(test.Street))
    street_train = lb_street.transform(train.Street)
    street_test = lb_street.transform(test.Street)

    lb_trap = LabelBinarizer().fit(list(train.Trap) + list(test.Trap))
    trap_train = lb_trap.transform(train.Trap)
    trap_test = lb_trap.transform(test.Trap)

    xs = np.hstack((train, spray_train, species_train,
                    street_train, trap_train))
    xs_sub = np.hstack((test, spray_test, species_test,
                        street_test, trap_test))

    m = StandardScaler()
    m.fit(xs)
    xs = m.transform(xs)
    xs_sub = m.transform(xs_sub)

    # add PCA
    from sklearn.decomposition import PCA
    m = PCA()
    m.fit(xs)
    xs_pca = m.transform(xs)
    xs_sub_pca = m.transform(xs_sub)

    xs = np.hstack((xs, xs_pca))
    xs_sub = np.hstack((xs_sub, xs_sub_pca))

    dump(xs, 'xs', True)
    dump(xs_sub, 'xs_sub', True)
    return xs, xs_sub


def xgb_parameter_search():
    from hyperopt import fmin, tpe, hp
    from kagura.xgbwrapper import XGBWrapper

    xs = load("xs")
    ys = load("ys")

    if args.tiny:
        tmp, xs, tmp, ys = stratified_split(xs, ys)

    train_xs, test_xs, train_ys, test_ys = stratified_split(xs, ys)

    def target_func((eta, max_depth, subsample, colsample_bytree)):
        global model
        model = XGBWrapper(
            eta=eta, max_depth=max_depth, test=(test_xs, test_ys),
            subsample=subsample, colsample_bytree=colsample_bytree,
            num_class=10
        )

        model.fit(train_xs, train_ys)
        log_loss = model.score(test_xs, test_ys)
        logging.info(
            "hyperopt eta=%f,max_depth=%d,subsample=%f"
            ",colsample_bytree=%f,log_loss=%f,best_iteration=%d",
            eta, max_depth, subsample, colsample_bytree,
            log_loss, model.bst.best_iteration)

        name = 'xgb_%f_%d_%f_%f_%f' % (eta, max_depth, subsample, colsample_bytree, log_loss)
        model.bst.save_model(name)
        return log_loss

    default_space = [
             hp.uniform('eta', 0, 1),
             hp.choice('max_depth', [4, 5, 6, 7, 8, 9]),
             hp.uniform('subsample', 0.4, 1),
             hp.uniform('colsample_bytree', 0.4, 1)]
    narrow_space = [
             hp.uniform('eta', 0.1, 0.4),
             hp.choice('max_depth', [5, 6]),
             hp.uniform('subsample', 0.8, 1),
             hp.uniform('colsample_bytree', 0.8, 1)]
    fmin(fn=target_func,
         space=narrow_space,
         algo=tpe.suggest,
         max_evals=10000)

    return


def make_submission():
    "出力された予測結果を提出用のCSVにする"
    import csv
    tmp, pred_sub, tmp = load_pawn(args.submit)
    # TODO アンサンブルの出力は上記形式ではないので対処が必要
    # アンサンブルの出力も同じ形式にすればよい

    name = 'submit_%s.csv' % args.name
    logging.info("writing submit_file to %s", name)
    fo = file(name, "w")
    fo.write(file('sampleSubmission.csv').readline())
    writer = csv.writer(fo)

    id = 1
    for p in pred_sub:
        if IS_BINARY_CLASSIFICATION:
            # Binary Classificationであるかどうかと出力がバイナリかは関係がない
            #p_bin = 1 if 0.5 <= p else 0
            #writer.writerow([id, p_bin])
            assert 0.0 <= p <= 1.0
            writer.writerow([id, "%0.5f" % p])
        else:
            assert all(0.0 <= v <= 1.0 for v in p)
            writer.writerow([id] + ["%0.5f" % v for v in p])

        id += 1
    fo.close()
    logging.info("wrote submit_file to %s", name)


def main():
    parser = get_default_arg_parser()
    parser.add_argument(
        '--boost', '-b', action='store',
        help='boost a result')

    parser.add_argument(
        '--random-boost', action='store_true',
        help='boost randomly')

    parser.add_argument(
        '--auto-boost', action='store_true',
        help='boost randomly')

    parser.add_argument(
        '--random-model', action='store_true',
        help='choose model randomly')

    parser.add_argument(
        '--random-ensemble', action='store_true',
        help='choose pawn output randomly and ensemble them')

    parser.add_argument(
        '--with-hole', action='store_true',
        help='avoid safety net')

    parser.add_argument(
        '-q', '--queen', action='store',
        help='specify pawns for the queen')

    parser.add_argument(
        '-p', '--pawn', action='store_true',
        help='make a pawn')

    args = get_args(parser)
    processqueue.listen(args)  # wait
    elapse = HumaneElapse()

    if args.with_hole:
        call = hole
    else:
        call = lambda f: f()

    if args.call_function:
        f = globals()[args.call_function]
        f()

    if args.submit:
        make_submission()

    if args.cross_validation:
        call(cross_validation)

    if args.pawn:
        call(cross_validation)

    if args.ensemble:
        call(cross_validation)

    if args.boost:
        call(cross_validation)

    if args.random_boost:
        choose_random_model()
        from glob import glob
        args.boost = choose_a_pawn()
        logging.info('boosting after: %s', args.boost)
        call(cross_validation)

    if args.auto_boost:
        auto_boost()

    if args.random_model:
        choose_random_model()
        call(cross_validation)

    if args.random_ensemble:
        choose_random_model()
        targets = choose_pawns()
        if len(targets) >= 2:
            args.ensemble = ','.join(targets)
            logging.info(
                'random ensemble: choose %d pawns',
                len(targets))
            logging.info('pawns: %s', args.ensemble)
            call(cross_validation)
        else:
            logging.info('cancel ensemble: not enough targets')

    if args.queen:
        queen()

    elapse.end()


def choose_random_model():
    "choose model and its parameters randomly"
    from random import choice
    args.model, args.param = choice([
#        ('Dummy', ''),
#        ('RF', '100'),
#        ('ExT', '20'),
#        ('ExT', choice('400 700 1000'.split())),
#        ('ExT_gini', choice('400 700 1000'.split())),
#        ('SVC', choice('rbf poly sigmoid'.split())),
#        ('KNN', ''),  # 間引かないと重たすぎる
#        ('SS_KNN', ''),  # 間引かないと重たすぎる
        ('XGBBin', choice('0.2 0.4 0.6 0.8'.split())),
#        ('NN', ''),
    ])
    make_better_name(args)
    logging.info('random model: %s', args.name)


def load_pawn(name):
    import os
    if os.path.exists(name):
        return load(name)
    n2 = 'pawn_' + name
    if os.path.exists(n2):
        return load(n2)
    import glob
    files = glob.glob('*%s*' % name)
    if len(files) == 1:
        return load(files[0])
    if len(files) > 1:
        preds = []
        pred_subs = []
        for target in files:
            pp, pps, tmp = load(target)
            preds.append(pp)
            pred_subs.append(pps)
        xs = np.hstack(preds)
        xs_sub = np.hstack(pred_subs)
        return xs, xs_sub, None
    raise RuntimeError('target %s is not found' % name)

def make_ensemble_input(targets):
    """
    load given pawns and do hstack them
    """
    preds = [get_train_data()]
    pred_subs = [get_test_data()]
    for target in targets.split(','):
        pp, pps, tmp = load_pawn(target)
        preds.append(pp)
        pred_subs.append(pps)
    xs = np.hstack(preds)
    ys = load('ys')
    xs_sub = np.hstack(pred_subs)
    return xs, ys, xs_sub


def cross_validation(tiny=False):
    logging.info("start cross validation")

    if args.boost:
        # 指定モデルでうまく分類できなかった半分を学習する
        # そのモデルの推測結果を加えてはいけない。
        # 正解の反転を教えることになる
        prev_pred, prev_pred_sub, prev_score = load_pawn(args.boost)

        xs = get_train_data()
        ys = load('ys')
        xs_sub = get_test_data()
    elif args.ensemble:
        xs, ys, xs_sub = make_ensemble_input(args.ensemble)

        # 束ねたものをそのまま特徴量にしても特にRF系では楽しくないので
        # PCAを掛けて回転してやる
        xs, xs_sub = whiten(xs, xs_sub)

    else:
        xs = get_train_data()
        ys = load('ys')
        xs_sub = get_test_data()
        prev_score = None

    if IS_BINARY_CLASSIFICATION:
        NUM_CLASSES = 1
    else:
        TODO  # 引数として取るか、設定を持つオブジェクトが必要

    NUM_TRAIN = len(xs)
    NUM_TEST = len(xs_sub)

    if args.tiny:  # 100分の1のデータ量にする
        tiny = range(0, len(xs), 100)
        xs = xs[tiny]
        ys = ys[tiny]
        if prev_score:
            prev_score = prev_score[tiny]
        xs_sub = xs_sub[:100, :]
        NUM_TRAIN = len(xs)
        NUM_TEST = len(xs_sub)

    if args.boost and prev_score != None:
        # 前回のスコア情報を用いて、自信のない50%に注力する
        median = np.median(prev_score)
        sure = (prev_score > median)[:, 0]
        not_sure = (prev_score <= median)[:, 0]
    else:
        sure = []
        not_sure = range(NUM_TRAIN)

    xs_sure = xs[sure]
    xs_notsure = xs[not_sure]
    ys_notsure = ys[not_sure]

    #skf = list(StratifiedKFold(ys_notsure, NUM_FOLDS))
    from sklearn.cross_validation import StratifiedShuffleSplit
    skf = list(StratifiedShuffleSplit(ys_notsure, NUM_FOLDS, random_state=12345))

    NUM_SURE = len(xs_sure)
    NUM_NOTSURE = len(xs_notsure)

    pred = np.zeros((NUM_TRAIN, NUM_CLASSES))
    pred_notsure = np.zeros((NUM_NOTSURE, NUM_CLASSES))
    pred_sure = np.zeros((NUM_SURE, NUM_CLASSES))
    pred_sure_for_bagging = np.zeros((NUM_SURE, NUM_FOLDS, NUM_CLASSES))
    pred_sub = np.zeros((NUM_TEST, NUM_CLASSES))
    pred_sub_for_bagging = np.zeros((NUM_TEST, NUM_FOLDS, NUM_CLASSES))
    scores = np.zeros(NUM_FOLDS)
    fit_times = np.zeros(NUM_FOLDS)
    pp_times = np.zeros(NUM_FOLDS)

    score_name = 'roc_auc'
    #score_name = 'log_loss'
    scorer = get_scorer(score_name)

    for i, (cvtrain, cvtest) in enumerate(skf):
        logging.info("Fold %d %s", i, args.name)
        xs_ns_cvtrain = xs_notsure[cvtrain]
        ys_ns_cvtrain = ys_notsure[cvtrain]
        xs_ns_cvtest = xs_notsure[cvtest]
        ys_ns_cvtest = ys_notsure[cvtest]

        model, pred_ns_cvtest, fit_time = fit_and_test(
            xs_ns_cvtrain, ys_ns_cvtrain,
            xs_ns_cvtest, ys_ns_cvtest
        )
        fit_times[i] = fit_time

        if IS_BINARY_CLASSIFICATION:
            pred_ns_cvtest = assure_col_vector(pred_ns_cvtest)

        pred_notsure[cvtest, :] = pred_ns_cvtest
        score = scorer(ys_ns_cvtest, pred_ns_cvtest)

        logging.info(
            "%s: %0.4f %s",
            score_name, score, args.name)
        scores[i] = score

        t = HumaneElapse('start prediction')
        pred_sub_by_this_model = model.predict_proba(xs_sub)
        if IS_BINARY_CLASSIFICATION:
            pred_sub_by_this_model = assure_col_vector(pred_sub_by_this_model)
        pred_sub_for_bagging[:, i, :] = pred_sub_by_this_model
        pp_times[i] = t.lap()
        logging.info("pp time: %s %s", t.get_human(), args.name)

        if args.boost:
            pred_sure_by_this_model = model.predict_proba(xs_sure)
            if IS_BINARY_CLASSIFICATION:
                pred_sure_by_this_model = assure_col_vector(pred_sure_by_this_model)
            pred_sure_for_bagging[:, i, :] = pred_sure_by_this_model

    # bagging
    pred_sub[:,:] = pred_sub_for_bagging.mean(1)
    pred_sure[:,:] = pred_sure_for_bagging.mean(1)
    # concat
    pred[sure ,:] = pred_sure
    pred[not_sure ,:] = pred_notsure
    pred_score = np.zeros((NUM_TRAIN, NUM_CLASSES))
    if IS_BINARY_CLASSIFICATION:
        num_t = float((ys == 1).sum())
        num_f = float((ys == 0).sum())
        assert num_t + num_f == NUM_TRAIN
        for i in range(len(ys)):
            if ys[i] == 0:
                # Fの時はそこより下のTの割合
                s = ys[pred[:, 0] < pred[i, 0]].sum() / num_t
            else:
                # Tの時はそこより上のFの割合
                s = (1 - ys)[pred[:, 0] > pred[i, 0]].sum() / num_f
            # が小さいほどよい
            pred_score[i] = -s
    else:
        pred_score = TODO

    name = 'pawn_%s' % args.name
    logging.info('writing data for queen: %s', name)
    dump((pred, pred_sub, pred_score), name)

    # 標準誤差の2倍がだいたい95%信頼区間
    B95 = 2 / np.sqrt(NUM_FOLDS)
    logging.info("Score(%s): %0.2f (+/- %0.2f)",
        score_name, scores.mean(), scores.std() * B95)
    logging.info("Fit time: %0.2f (+/- %0.2f)",
        fit_times.mean(), fit_times.std() * B95)
    logging.info("Predict time: %0.2f (+/- %0.2f)",
        pp_times.mean(), pp_times.std() * B95)

    # digested one line info
    parent = 'None'
    if args.boost: parent = args.boost
    if args.ensemble: parent = args.ensemble
    total_score = scorer(ys, pred)
    logging.info(
        "PAWN model=%s(%s) score=%0.2f (+/- %0.2f), total_score=%0.2f",
        args.model, args.param, scores.mean(), scores.std() * B95, total_score)
    logging.info("PAWN parent=%s", parent)
    logging.info("finish cross validation")



def queen():
    from sklearn.preprocessing import StandardScaler
    #from glob import glob
    #targets = glob('pawn_*')
    #targets = file('pawns2.txt').readlines()

    xs, ys, xs_sub = make_ensemble_input(args.queen)

    #s = StandardScaler()
    #s.fit(xs)
    #xs = s.transform(xs)
    #xs_sub = s.transform(xs_sub)
    xs, xs_sub = whiten(xs, xs_sub)

    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression()
    #from sklearn.ensemble import RandomForestClassifier
    #m = RandomForestClassifier(
    #    400, criterion='entropy')
    m.fit(xs, ys)
    pred_sub = m.predict_proba(xs_sub)

    name = 'queen_%s' % args.name
    dump((None, pred_sub[:, [0]], None), name)

    I = np.eye(len(xs[0]))
    pp = m.predict_proba(I)
    dump(pp, 'pp')


def get_scorer(name):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import log_loss
    scorer = {
        'roc_auc': roc_auc_score,
        'log_loss': log_loss
    }[name]
    return scorer


def choose_a_pawn():
    from random import choice
    from glob import glob
    return  choice(glob('pawn_*'))


def auto_boost():
    import subprocess
    i = 1
    # to shutdown gracefully,
    # use SIGUSR1 and set args.auto_boost=False
    ME = ['./westnile.py', '--with-hole']
    nameopt = "--name=%s_%%d" % args.name

    BOOST = ['--random-boost --with-hole']
    ENSEMBLE = 'python westnile.py --random-ensemble --with-hole'

    while args.auto_boost:
        logging.info('auto boosting %d', i)
        if np.random.random() < 0.5:
            command = '--random-boost'
        else:
            command = '--random-ensemble'

        subprocess.call(
            ME + [command, nameopt % i])
        i += 1


def choose_pawns(p=0.5):
    from glob import glob
    targets = glob('pawn_*')
    N = len(targets)
    targets = [x for x in targets
               if np.random.random() < p]
    return targets


def whiten(xs, xs_sub):
    "given xs and xs_sub, whiten them with PCA"
    from sklearn.decomposition import PCA
    logging.info("PCA transforms %d dim inputs", len(xs[0]))
    m = PCA(n_components=len(xs[0]), whiten=True)
    m.fit(xs)
    xs = m.transform(xs)
    xs_sub = m.transform(xs_sub)
    return xs, xs_sub


def assure_col_vector(a):
    if len(a.shape) == 1:
        a = a.reshape(len(a), 1)
    elif a.shape[1] == 2:
        a = a[:, [1]]
    return a


def fit_and_test(xs_train, ys_train, xs_test, ys_test):
    model = get_model(args)
    if hasattr(model, 'fit_and_test'):
        return model.fit_and_test(
            xs_train, ys_train, xs_test, ys_test)
    t = HumaneElapse('start fitting')
    model.fit(xs_train, ys_train)
    fit_time = t.lap()
    logging.info(
        "fit time: %s %s",
        t.get_human(), args.name)

    pred_test = model.predict_proba(xs_test)
    return model, pred_test, fit_time


def init_dataset():
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import StratifiedShuffleSplit
    data = load_digits()
    data.target = (data.target == 5)  # Binary Classification
    train, test = list(StratifiedShuffleSplit(data.target, n_iter=1, test_size=0.5))[0]
    xs = data.data[train]
    ys = data.target[train]
    xs_sub = data.data[test]
    ys_sub = data.target[test]
    dump(xs, 'xs')
    dump(ys, 'ys')
    dump(xs_sub, 'xs_sub')
    dump(ys_sub, 'ys_sub')

#///
if __name__ == "__main__":
    safetynet(main)
