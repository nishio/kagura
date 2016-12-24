
# -*- encoding: utf-8 -*-
"""
クロスバリデーション。再利用するのでとりあえずここに置いたけど
他の関数への依存がヘビーなのでこのままでは動かない。
整理が必要。
"""

from sklearn import cross_validation
NUM_FOLDS = 5

def cross_validation(tiny=False):
    logging.info("start cross validation")
    if args.train_data.startswith('blend'):
        # blendから読む
        blend_train, blend_test = get_blend(args.train)
        ys = load('ys')
        xs = blend_train
        xs_sub = blend_test
    else:
        xs, ys = get_train_data()
        xs_sub = get_test_data()

    model = get_model(args)

    if args.tiny:  # 100分の1のデータ量にする
        tiny = range(0, len(xs), 100)
        xs = xs[tiny]
        ys = ys[tiny]
        xs_sub = xs_sub[:100, :]


    skf = list(cross_validation.StratifiedKFold(ys, NUM_FOLDS))


    logging.info(model)
    blend_train = np.zeros((NUM_TRAIN, NUM_CLASSES))
    blend_test = np.zeros((NUM_TEST, NUM_CLASSES))
    blend_test_j = np.zeros((NUM_TEST, NUM_FOLDS, NUM_CLASSES))
    scores = np.zeros(NUM_FOLDS)
    fit_times = np.zeros(NUM_FOLDS)
    pp_times = np.zeros(NUM_FOLDS)
    from sklearn.metrics.scorer import check_scoring
    scorer = check_scoring(model, 'log_loss')

    for i, (train, test) in enumerate(skf):
        logging.info("Fold %d %s", i, args.name)
        X_train = xs[train]
        y_train = ys[train]
        X_test = xs[test]
        y_test = ys[test]

        t = time.time()
        model.fit(X_train, y_train)
        fit_times[i] = time.time() - t
        logging.info(
            "fit time: %.2f %s",
            fit_times[i], args.name)

        P = model.predict_proba(X_test)
        score = scorer(model, X_test, y_test)
        scores[i] = score
        logging.info("score: %0.4f %s", score, args.name)

        t = time.time()
        blend_test_j[:, i, :] = model.predict_proba(xs_sub)
        pp_times[i] = time.time() - t

        logging.info("pp time: %f %s", pp_times[i], args.name)

        blend_train[test, :] = P

    blend_test[:,:] = blend_test_j.mean(1)
    name = 'blend_%s' % args.name
    logging.info('writing data for blending: %s', name)
    dump((blend_train, blend_test), name)

    # 標準誤差の2倍がだいたい95%信頼区間
    B95 = 2 / np.sqrt(NUM_FOLDS)
    logging.info("Log Loss: %0.2f (+/- %0.2f)",
        scores.mean(), scores.std() * B95)
    logging.info("Fit time: %0.2f (+/- %0.2f)",
        fit_times.mean(), fit_times.std() * B95)
    logging.info("Predict time: %0.2f (+/- %0.2f)",
        pp_times.mean(), pp_times.std() * B95)
    logging.info("finish cross validation")
