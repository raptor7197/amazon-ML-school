import argparse, os, json, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from scipy.optimize import minimize
import scipy.sparse as sp
import lightgbm as lgb
import xgboost as xgb

from src.features import prepare
from src.smape import smape_numpy

def bins_for_stratify(y, n=10):
    return pd.qcut(y, q=n, labels=False, duplicates="drop")

def adv_weights(train_df, test_df, vec):
    from sklearn.linear_model import LogisticRegression
    Xtr = vec.transform(train_df["text"])
    Xte = vec.transform(test_df["text"])
    X = sp.vstack([Xtr, Xte])
    y = np.array([0]*Xtr.shape[0] + [1]*Xte.shape[0])
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    p = clf.predict_proba(Xtr)[:,1]
    w = 1.0 / np.clip(1.0 - p, 1e-3, 1.0)
    w = np.clip(w / (w.mean() + 1e-12), 0.2, 5.0)
    return w

def blend(w, arrs):
    w = np.clip(np.asarray(w), 0, 1)
    w = w / (w.sum() + 1e-12)
    out = np.zeros_like(arrs[0])
    for wi, ai in zip(w, arrs):
        out += wi * ai
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tr = pd.read_csv(args.train_csv)
    te = pd.read_csv(args.test_csv)

    tr = prepare(tr); te = prepare(te)

    word = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.995, max_features=500000, strip_accents="unicode", sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3,6), min_df=3, max_features=250000, sublinear_tf=True)

    Xw_tr = word.fit_transform(tr["text"])
    Xc_tr = char.fit_transform(tr["text"])
    X_tr = sp.hstack([Xw_tr, Xc_tr]).tocsr()

    Xw_te = word.transform(te["text"])
    Xc_te = char.transform(te["text"])
    X_te = sp.hstack([Xw_te, Xc_te]).tocsr()

    svd = TruncatedSVD(n_components=300, random_state=42)
    Z_tr = svd.fit_transform(X_tr)
    Z_te = svd.transform(X_te)

    num_cols = ["pack","qty","ipq","len","digits"]
    ss = StandardScaler()
    N_tr = ss.fit_transform(tr[num_cols].values)
    N_te = ss.transform(te[num_cols].values)

    w_adv = adv_weights(tr, te, word)

    y = np.log1p(tr["price"].values.astype(float))
    bins = bins_for_stratify(tr["price"].values, 10)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    oof_r, oof_e, oof_h = np.zeros(tr.shape[0]), np.zeros(tr.shape[0]), np.zeros(tr.shape[0])
    oof_l30, oof_l50, oof_l70 = np.zeros(tr.shape[0]), np.zeros(tr.shape[0]), np.zeros(tr.shape[0])
    oof_x = np.zeros(tr.shape[0])

    p_r, p_e, p_h = np.zeros(te.shape[0]), np.zeros(te.shape[0]), np.zeros(te.shape[0])
    p_l30, p_l50, p_l70 = np.zeros(te.shape[0]), np.zeros(te.shape[0]), np.zeros(te.shape[0])
    p_x = np.zeros(te.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(skf.split(tr, bins), 1):
        Xtr_b = sp.hstack([X_tr[tr_idx], Z_tr[tr_idx]])
        Xva_b = sp.hstack([X_tr[va_idx], Z_tr[va_idx]])
        Xte_b = sp.hstack([X_te, Z_te])
        Ntr, Nva = N_tr[tr_idx], N_tr[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        wtr = w_adv[tr_idx]

        R = Ridge(alpha=2.0, random_state=42)
        R.fit(sp.hstack([Xtr_b, Ntr]), ytr, sample_weight=wtr)
        oof_r[va_idx] = R.predict(sp.hstack([Xva_b, Nva])); p_r += R.predict(sp.hstack([Xte_b, N_te]))/args.folds

        E = ElasticNet(alpha=0.02, l1_ratio=0.12, max_iter=20000, random_state=42)
        E.fit(sp.hstack([Xtr_b, Ntr]), ytr, sample_weight=wtr)
        oof_e[va_idx] = E.predict(sp.hstack([Xva_b, Nva])); p_e += E.predict(sp.hstack([Xte_b, N_te]))/args.folds

        H = HuberRegressor(epsilon=1.3, alpha=1e-4, max_iter=2000)
        H.fit(sp.hstack([Xtr_b, Ntr]).toarray(), ytr)
        oof_h[va_idx] = H.predict(sp.hstack([Xva_b, Nva]).toarray()); p_h += H.predict(sp.hstack([Xte_b, N_te]).toarray())/args.folds

        for q, oof_buf, p_buf in [(0.3, oof_l30, p_l30), (0.5, oof_l50, p_l50), (0.7, oof_l70, p_l70)]:
            dtr = lgb.Dataset(sp.hstack([Xtr_b, Ntr]), label=ytr, weight=wtr)
            dva = lgb.Dataset(sp.hstack([Xva_b, Nva]), label=yva, reference=dtr)
            params = dict(objective="quantile", alpha=q, learning_rate=0.05, num_leaves=127, feature_fraction=0.85,
                          bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=20, max_depth=-1, verbose=-1, seed=41+fold)
            GBM = lgb.train(params, dtr, num_boost_round=7000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
            oof_buf[va_idx] = GBM.predict(sp.hstack([Xva_b, Nva]), num_iteration=GBM.best_iteration)
            p_buf += GBM.predict(sp.hstack([Xte_b, N_te]), num_iteration=GBM.best_iteration)/args.folds

        dtr = xgb.DMatrix(sp.hstack([Xtr_b, Ntr]), label=ytr, weight=wtr)
        dva = xgb.DMatrix(sp.hstack([Xva_b, Nva]), label=yva)
        dte = xgb.DMatrix(sp.hstack([Xte_b, N_te]))
        params_x = dict(objective="reg:tweedie", tweedie_variance_power=1.2, eta=0.05, max_depth=8, subsample=0.85,
                        colsample_bytree=0.85, min_child_weight=1.0, reg_lambda=1.0, nthread=0, seed=45+fold)
        XGB = xgb.train(params_x, dtr, num_boost_round=7000, evals=[(dva,"valid")], early_stopping_rounds=200, verbose_eval=False)
        oof_x[va_idx] = XGB.predict(dva, iteration_range=(0, XGB.best_iteration+1))
        p_x += XGB.predict(dte, iteration_range=(0, XGB.best_iteration+1))/args.folds

    oof_list = [np.expm1(oof_r), np.expm1(oof_e), np.expm1(oof_h),
                np.expm1(oof_l30), np.expm1(oof_l50), np.expm1(oof_l70), np.expm1(oof_x)]
    te_list  = [np.expm1(p_r),  np.expm1(p_e),  np.expm1(p_h),
                np.expm1(p_l30), np.expm1(p_l50), np.expm1(p_l70), np.expm1(p_x)]

    y_true = tr["price"].values.astype(float)

    def fobj(w):
        return smape_numpy(y_true, np.clip(blend(w, oof_list), 0.01, None))

    res = minimize(fobj, x0=np.ones(len(oof_list)), method="L-BFGS-B", bounds=[(0,1)]*len(oof_list))
    w = np.clip(res.x, 0, 1); w = w / (w.sum() + 1e-12)

    meta_X = np.vstack(oof_list).T
    meta_te = np.vstack(te_list).T
    Rm = Ridge(alpha=1.0, random_state=42)
    Rm.fit(meta_X, np.log1p(y_true))
    te_meta = np.expm1(Rm.predict(meta_te))

    pred = 0.5*blend(w, te_list) + 0.5*te_meta

    tmp = pd.DataFrame({"brand": tr["brand"], "qty_bucket": tr["qty_bucket"], "y": y_true, "p": blend(w, oof_list)})
    g = tmp.groupby(["brand","qty_bucket"]).agg(y_med=("y","median"), p_med=("p","median"), n=("y","size")).reset_index()
    g["ratio"] = g["y_med"] / np.clip(g["p_med"], 1e-6, None)
    g["shrink"] = 1 / (1 + np.exp(-(g["n"]-10)/5))
    g["adj"] = (1 + g["shrink"] * (g["ratio"] - 1)).clip(0.6, 1.6)
    key = pd.merge(te[["brand","qty_bucket"]], g[["brand","qty_bucket","adj"]], how="left", on=["brand","qty_bucket"])
    pred = np.clip(pred * key["adj"].fillna(1.0).values, 0.01, None)

    sub = pd.DataFrame({"sample_id": te["sample_id"], "price": pred})
    sub.to_csv(args.out_csv, index=False)

    meta = {"weights": w.tolist(), "oof_smape": float(fobj(w))}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("weights:", w)
    print("OOF SMAPE:", meta["oof_smape"])
    print("saved:", args.out_csv)

if __name__ == "__main__":
    main()