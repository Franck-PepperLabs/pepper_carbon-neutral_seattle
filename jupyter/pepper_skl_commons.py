# Scikit-Learn commons
from pepper_commons import *


"""
Preprocessing : split and scale dataset
"""
def last_s_u(ml_data):
    """Return the last column label of format s_u_{indice}, which is the last feature column"""
    return ml_data.columns[ml_data.columns.str.startswith('s_u')][-1]

def get_X_Y(ml_data):
    """Return the couple features (X) and targets (Y)"""
    return ml_data.loc[:, 'bid':last_s_u(ml_data)], ml_data.loc[:, 'ies_wn':]

from sklearn import preprocessing
def scale_X(X_train, X_test):
    """Return scaled X_train and X_test"""
    std_scale = preprocessing.StandardScaler().fit(X_train)
    std_X_train = pd.DataFrame(std_scale.transform(X_train), columns=X_train.columns)
    std_X_test = pd.DataFrame(std_scale.transform(X_test), columns=X_test.columns)
    return std_X_train, std_X_test

from sklearn import model_selection
class Dataset:

    def __init__(self, data, name, random_state, test_size):
        self.data = data
        self.name = name
        self.random_state = random_state
        self.test_size = test_size
        X, Y = get_X_Y(data)
        self.features = list(X.columns)
        self.targets = list(Y.columns)
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(
            X, Y, random_state=random_state, test_size=test_size)
        self._X_train, self._X_test = scale_X(self.X_train, self.X_test)
        self.X = self.X_train, self.X_test
        self._X = self._X_train, self._X_test
        self.XY = self.X_train, self.X_test, self.Y_train, self.Y_test
        self._XY = self._X_train, self._X_test, self.Y_train, self.Y_test

    def __str__(self):
        self_str = f'{self.name} | seed: {self.random_state} | '
        self_str += f'train-test: {round(100 * (1 - self.test_size))}-{round(100 * self.test_size)}\n'
        self_str += f'features: {self.features}\n'
        self_str += f'targets: {self.targets}\n'

        return self_str


"""
Evaluation of performances
"""




"""
Baseline dummies
"""

def get_baseline_err(X_train, y_train, X_test, y_test):
    # baseline
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    # erreur **RSS** (*Residual Sum of Squares*)
    baseline_err = np.mean((lr.predict(X_test) - y_test) ** 2)
    return baseline_err


from sklearn import linear_model
from sklearn.model_selection import cross_val_score
def get_lr_baseline_err(X_train, y_train):
    lr = linear_model.LinearRegression()
    lr_reg = lr.fit(X_train, y_train)
    return cross_val_score(lr_reg, X_train, y_train, cv=5).mean()

"""
Grid search of best (hyper)parameters
"""

# premi??re version sp??cialis??e Ridge
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
def best_ridge_alpha(X_train, X_test, y_train, y_test, baseline_err, n_alphas = 50):
    alphas = np.logspace(-5, 5, n_alphas)   # distribution logarithmique entre 10^-5 et 10^5

    ridge = linear_model.Ridge()

    coefs, errors = [], []
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X_train, y_train)
        coefs.append(ridge.coef_)
        errors.append(np.mean((ridge.predict(X_test) - y_test) ** 2))

    ax = plt.gca()
    ax.plot(alphas, errors, [10**-5, 10**5], [baseline_err, baseline_err])
    ax.set_xscale('log')
    plt.show()

    # meilleur param??tre
    i = np.argmin(errors)
    print('best param : argmin', i, '=> error', errors[i], '=> alpha', alphas[i])

    # chemin de r??gularisation
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.show()

    return alphas[i]


# premi??re version sp??cialis??e Lasso
from sklearn.linear_model import LassoCV
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def best_lasso_alpha(X, y): # X_train, X_test, y_train, y_test, baseline_err):
    # Let???s start by making the hyperparameter tuning using LassoCV.

    start_time = time.time()
    model = make_pipeline(StandardScaler(), LassoCV(cv=20)).fit(X, y)
    fit_time = time.time() - start_time

    import matplotlib.pyplot as plt

    #ymin, ymax = 0, 20000
    lasso = model[-1]
    plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
    plt.plot(
        lasso.alphas_,
        lasso.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

    #plt.ylim(ymin, ymax)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Mean square error")
    plt.legend()
    _ = plt.title(
        f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
    )
    plt.show()

    return lasso.alpha_

def builtin_best_lasso_alpha(X_train, y_train):
    lasso = linear_model.LassoCV(cv=10, random_state=42)
    start_time = time.time()
    lasso.fit(X_train, np.ravel(y_train))
    fit_time = time.time() - start_time
    return lasso, fit_time

def builtin_best_elastic_alpha(X_train, y_train):
    elastic = linear_model.ElasticNetCV(cv=10, random_state=42, l1_ratio=[.5, .7, .8, .85, .9, .95])
    start_time = time.time()
    elastic.fit(X_train, np.ravel(y_train))
    fit_time = time.time() - start_time
    return elastic, fit_time


def show_builtin_best_lasso_alpha(lasso, Xy, X, y):
    X_train, X_test, y_train, y_test = Xy
    print('best', bold('alpha'), ':', lasso.alpha_)
    print('    train', bold('score'), ':', lasso.score(X_train, y_train))
    print('     test', bold('score'), ':', lasso.score(X_test, y_test))
    cv_scores = cross_val_score(lasso, X, np.ravel(y), cv=3)
    print('3-CV mean', bold('score'), ':', np.mean(cv_scores))
    print(' 3-CV std', bold('score'), ':', np.std(cv_scores))

def show_builtin_best_elastic_cv_result(elastic, Xy, X, y):
    X_train, X_test, y_train, y_test = Xy
    print('best', bold('alpha'), ':', elastic.alpha_)
    print('best', bold('l1_ratio'), ':', elastic.l1_ratio_)
    print('    train', bold('score'), ':', elastic.score(X_train, y_train))
    print('     test', bold('score'), ':', elastic.score(X_test, y_test))
    cv_scores = cross_val_score(elastic, X, np.ravel(y), cv=3)
    print('3-CV mean', bold('score'), ':', np.mean(cv_scores))
    print(' 3-CV std', bold('score'), ':', np.std(cv_scores))


import matplotlib.pyplot as plt
def plot_builtin_best_lasso_alpha(lasso, fit_time):
    plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
    plt.plot(
        lasso.alphas_,
        lasso.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

    #plt.ylim(ymin, ymax)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Mean square error")
    plt.legend()
    _ = plt.title(
        f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
    )
    plt.show()

import matplotlib.pyplot as plt
def plot_builtin_best_elastic_alpha(elastic, fit_time):
    for k, l1r in enumerate([.5, .7, .8, .85, .9, .95]):
        plt.semilogx(elastic.alphas_[k], elastic.mse_path_[k], linestyle=":")
        plt.plot(
            elastic.alphas_[k],
            elastic.mse_path_[k].mean(axis=-1),
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(elastic.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

        #plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean square error")
        plt.legend()
        _ = plt.title(
            f"Mean square error on each fold with l1_ratio {l1r} : coordinate descent (train time: {fit_time:.2f}s)"
        )
        plt.show()


def builtin_best_lasso_cv_search(Xy, X, y):
    X_train, _, y_train, _ = Xy
    lasso, fit_time = builtin_best_lasso_alpha(X_train, y_train)
    show_builtin_best_lasso_alpha(lasso, Xy, X, y)
    plot_builtin_best_lasso_alpha(lasso, fit_time)
    return lasso

def builtin_best_elastic_cv_search(Xy, X, y):
    X_train, _, y_train, _ = Xy
    elastic, fit_time = builtin_best_elastic_alpha(X_train, y_train)
    show_builtin_best_elastic_cv_result(elastic, Xy, X, y)
    plot_builtin_best_elastic_alpha(elastic, fit_time)
    return elastic

import numpy as np
def get_best_params(Xy, model, param_grid, baseline_err, cv=10, verbose=False):
    X_train, _, y_train, _ = Xy
    if verbose:
        print('Searching best params among :', param_grid)

    verbosity = 3 if verbose else 0
    gs = model_selection.GridSearchCV(model, param_grid, cv=cv, verbose=verbosity) #, scoring='r2', refit='r2')
    print('coucou', end='')
    gs.fit(X_train, np.ravel(y_train))
    print('kookoo')

    cv_res = pd.DataFrame.from_dict(gs.cv_results_)
    if verbose:
        print(gs.best_params_)
        # print('Best param (r2 score) :', round(cv_res.mean_test_r2[cv_res.rank_test_r2 == 1].values[0], 2))
        print('Cross validation results :')
        display(cv_res)

    return gs.best_estimator_, gs.best_params_, gs.best_score_, gs.best_index_, gs.scorer_, cv_res
    #return gs.best_params_


def show_best_params(gbp_res, Xy, X, y):
    print_subtitle("Best")
    print('best', bold('estimator'), ':', gbp_res[0])
    print('best', bold('params'), ':', gbp_res[1])
    print('best', bold('score'), ':', gbp_res[2])
    print('best', bold('index'), ':', gbp_res[3])

    print_subtitle("Scores")
    best_estimator = gbp_res[0]
    X_train, X_test, y_train, y_test = Xy
    print('    train', bold('score'), ':', best_estimator.score(X_train, y_train))
    print('     test', bold('score'), ':', best_estimator.score(X_test, y_test))
    cv_scores = cross_val_score(best_estimator, X, y, cv=3)
    print('3-CV mean', bold('score'), ':', np.mean(cv_scores))
    print(' 3-CV std', bold('score'), ':', np.std(cv_scores))


def select_important_features(estimator, eps=0):
    eps = 0
    bindex = np.abs(estimator.coef_) > eps
    n_if = bindex.sum()
    important_features = ['__const__'] + list(estimator.feature_names_in_[bindex])
    important_coefs = list(estimator.intercept_) + list(estimator.coef_[bindex])
    print(f"There are {n_if} important features")
    coefs = pd.DataFrame(important_coefs, index=important_features, columns=['coef'])
    return coefs


import matplotlib.pyplot as plt
def show_alpha_path(cv_res, best_params, baseline_err, min_alpha_log, max_alpha_log, pfx=''):
    a_key = pfx + 'alpha'
    pa_key = 'param_' + a_key
    ax = plt.gca()
    ax.plot(cv_res[pa_key], cv_res.mean_test_score,
            [10**min_alpha_log, 10**max_alpha_log], [baseline_err, baseline_err])
    ax.set_xscale('log')
    ax.vlines(
        best_params[a_key],
        cv_res.mean_test_score.min(),
        cv_res.mean_test_score.max(),
        color="black",
        linestyle="--",
        label="Best alpha",
    )
    plt.legend()
    plt.show()



"""
Evaluation : Performances analysis and scores saving
"""
from sklearn import metrics
def print_perf_measures(y_real, y_pred):
    measures = {
        'RMSE': metrics.mean_squared_error(y_real, y_pred),
        'R2': metrics.r2_score(y_real, y_pred)
    }
    print(bold('performances'), ':')
    for k, m in measures.items():
        print(f'\t{k} :', round(m, 2))


# seconde version avec m??morisation des scores
import pandas as pd
def empty_scores():
    return pd.DataFrame(
        #columns=['date', 'method', 'params', 'dataset', 'target', 'dummy', 'train_score', 'test_score', 'r2', 'dummy_r2']
        columns=[
            'date', 'dataset', 'target', 'method', 'params', 'dummy',
            'train_eval_time', 'train_score', 'test_score', 'dummy_r2']
    )

def append_score(scores, score):
    id = scores.shape[0]
    scores.loc[id] = score


from sklearn import linear_model
def test_append_scores():
    scores = empty_scores()
    now_ts = pd.Timestamp.now()
    meth = linear_model.Ridge
    score = [now_ts, meth, '_', '_', '_', '_', '_', '_', '_', '_']
    append_score(scores, score)
    append_score(scores, score)
    display(scores)

from sklearn import metrics

def unpack_item(s, key):
    return s.apply(lambda x: x[key]).values

# graphique 3D score = f(random_state, test_size)
def surf3d_rs_ts_tsc(scores, label):
    #scores = all_scores[all_scores.dataset == dataset_label]

    x = unpack_item(scores.params, 'random_state')
    y = unpack_item(scores.params, 'test_size')

    # on r??duit les scores ?? l'intervalle -1, 1 : les r??sultat contreperformants < -.25 sont ramen??s ?? -.25
    test_scores = scores.test_score.copy()
    test_scores[test_scores < -.25] = -.25
    z = test_scores.values

    # voir https://stackoverflow.com/questions/21161884/plotting-a-3d-surface-from-a-list-of-tuples-in-matplotlib
    #from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    import numpy as np

    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(projection="3d")
    fig.add_axes(ax)

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.plasma)

    ax.set_xlabel('random_state')
    ax.set_ylabel('test_size')
    ax.set_zlabel('test_score')

    # pretty init view
    ax.view_init(elev=22, azim=110)
    plt.colorbar(surf)
    plt.suptitle(f'{label} test scores depending on random_state and test_size')
    plt.subplots_adjust(top=0.9)

    plt.show()


# graphique 3D score = f(p1, p2)
def surf3d_score_p1_p2(cv_results, tgt_label, param_1st, param_2nd):
    #scores = all_scores[all_scores.dataset == dataset_label]

    x = cv_results[param_1st]   # unpack_item(scores.params, 'random_state')
    y = cv_results[param_2nd]   # unpack_item(scores.params, 'test_size')

    # on r??duit les scores ?? l'intervalle -1, 1 : les r??sultat contreperformants < -.25 sont ramen??s ?? -.25
    test_scores = scores['mean_test_score'].copy()
    test_scores[test_scores < -.25] = -.25
    z = test_scores.values

    # voir https://stackoverflow.com/questions/21161884/plotting-a-3d-surface-from-a-list-of-tuples-in-matplotlib
    #from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    import numpy as np

    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(projection="3d")
    fig.add_axes(ax)

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.plasma)

    ax.set_xlabel(param_1st)
    ax.set_ylabel(param_2nd)
    ax.set_zlabel('test_score')

    # pretty init view
    ax.view_init(elev=22, azim=110)
    plt.colorbar(surf)
    plt.suptitle(f'{tgt_label} test scores depending on {param_1st} and {param_2nd}')
    plt.subplots_adjust(top=0.9)

    plt.show()

"""
From SKL User Guide
"""

# 3.1.1 Ex Plotting Cross-Validated Predictions
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-auto-examples-model-selection-plot-cv-predict-py

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
def plot_cross_val_predict(est, X, y):
    #est : lr = linear_model.LinearRegression()
    #X, y = datasets.load_diabetes(return_X_y=True)

    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(est, X, y, cv=10)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()


# 1.1.3 Lasso model selection: AIC-BIC / cross-validation
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py


import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
@ignore_warnings(category=ConvergenceWarning)
def lasso_best_from_ic(X, y, show_tab=False, show_plot=False):
    # We will first fit a Lasso model with the AIC criterion.
    start_time = time.time()
    lasso_lars_ic = make_pipeline(
        StandardScaler(), LassoLarsIC(criterion="aic", normalize=False)
    ).fit(X, y)
    fit_time = time.time() - start_time

    # We store the AIC metric for each value of alpha used during fit.
    results = pd.DataFrame(
        {
            "alphas": lasso_lars_ic[-1].alphas_,
            "AIC criterion": lasso_lars_ic[-1].criterion_,
        }
    ).set_index("alphas")
    alpha_aic = lasso_lars_ic[-1].alpha_

    # Now, we perform the same analysis using the BIC criterion.
    lasso_lars_ic.set_params(lassolarsic__criterion="bic").fit(X, y)
    results["BIC criterion"] = lasso_lars_ic[-1].criterion_
    alpha_bic = lasso_lars_ic[-1].alpha_

    # We can check which value of alpha leads to the minimum AIC and BIC.
    if show_tab:
        def highlight_min(x):
            x_min = x.min()
            return ["font-weight: bold" if v == x_min else "" for v in x]

        display(results.style.apply(highlight_min))

    # Finally, we can plot the AIC and BIC values for the different alpha values.
    # The vertical lines in the plot correspond to the alpha chosen for each criterion.
    # The selected alpha corresponds to the minimum of the AIC or BIC criterion.
    if show_plot:
        ax = results.plot()
        ax.vlines(
            alpha_aic,
            results["AIC criterion"].min(),
            results["AIC criterion"].max(),
            label="alpha: AIC estimate",
            linestyles="--",
            color="tab:blue",
        )
        ax.vlines(
            alpha_bic,
            results["BIC criterion"].min(),
            results["BIC criterion"].max(),
            label="alpha: BIC estimate",
            linestyle="--",
            color="tab:orange",
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("criterion")
        ax.set_xscale("log")
        ax.legend()
        _ = ax.set_title(
            f"Information-criterion for model selection (training time {fit_time:.2f}s)"
        )
        plt.show()

    print(bold('alpha: AIC estimate'), ':', alpha_aic)
    print(bold('alpha: BIC estimate'), ':', alpha_bic)
    # print(bold('training time'), ':', round(fit_time, 3))

    return results, alpha_aic, alpha_bic, fit_time


from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
def lasso_best_from_cd_cv(X, y, show_tab=False, show_plot=False):
    # Let???s start by making the hyperparameter tuning using LassoCV.
    start_time = time.time()
    model = make_pipeline(StandardScaler(), LassoCV(cv=20)).fit(X, y)
    fit_time = time.time() - start_time
    lasso = model[-1]

    if show_plot:
        #ymin, ymax = 2300, 3800
        plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
        plt.plot(
            lasso.alphas_,
            lasso.mse_path_.mean(axis=-1),
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

        #plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean square error")
        plt.legend()
        _ = plt.title(
            f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
        )
        plt.show()

    print(bold('alpha: CD CV estimate'), ':', lasso.alpha_)

    return lasso.alpha_, lasso.mse_path_.mean(axis=-1), lasso.alphas_, lasso.mse_path_, fit_time


from sklearn.linear_model import LassoLarsCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
@ignore_warnings(category=ConvergenceWarning)
def lasso_best_from_lars_cv(X, y, show_tab=False, show_plot=False):
    # Let???s start by making the hyperparameter tuning using LassoLarsCV.
    start_time = time.time()
    model = make_pipeline(StandardScaler(), LassoLarsCV(cv=20, normalize=False)).fit(X, y)
    fit_time = time.time() - start_time
    lasso = model[-1]

    if show_plot:
        plt.semilogx(lasso.cv_alphas_, lasso.mse_path_, ":")
        plt.semilogx(
            lasso.cv_alphas_,
            lasso.mse_path_.mean(axis=-1),
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha CV")

        #plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean square error")
        plt.legend()
        _ = plt.title(f"Mean square error on each fold: Lars (train time: {fit_time:.2f}s)")
        plt.show()

    print(bold('alpha: LARS VC estimate'), ':', lasso.alpha_)

    return lasso.alpha_, lasso.mse_path_.mean(axis=-1), lasso.alphas_, lasso.mse_path_, fit_time


from pepper_commons import print_title, print_subtitle, bold  # pretty print
"""from pepper_skl_commons import (
    lasso_best_from_ic,
    lasso_best_from_cd_cv,
    lasso_best_from_lars_cv
)"""
import numpy as np

def lasso_selection(X, y):
    best_alphas = dict()
    _y = np.ravel(y)
    
    print_subtitle(f'Lasso IC selection')
    (
        _, best_alphas['aic'], best_alphas['bic'], _
        # _ ??? results, alpha_aic, alpha_bic, fit_time
    ) = lasso_best_from_ic(X, _y, show_tab=False, show_plot=True)
    best_alphas['mic'] = (best_alphas['aic'] + best_alphas['bic']) / 2
    print(bold('alpha: AIC / BIC mean estimate'), ':', best_alphas['mic'])

    print_subtitle(f'Lasso CD CV selection')
    (
        best_alphas['cd_cv'], _, _, _, _
        # _ ???  alpha_, mse_path_mean, alphas_, mse_path_, fit_time
    ) = lasso_best_from_cd_cv(X, _y, show_tab=False, show_plot=True)

    print_subtitle(f'Lasso LARS CV selection')
    (
        best_alphas['lars_cv'], _, _, _, _
        # _ ???  alpha_, mse_path_mean, alphas_, mse_path_, fit_time
    ) = lasso_best_from_lars_cv(X, _y, show_tab=False, show_plot=True)

    print_subtitle(f'Synthesis')
    display(best_alphas)
    mean_estimate = (best_alphas['mic'] + best_alphas['cd_cv'] + best_alphas['lars_cv']) / 3
    print(bold('alpha: mean estimate'), ':', mean_estimate)
    return best_alphas, mean_estimate


# source : https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path

def plot_lasso_coordinate_descent_path(X, y):
    y = np.ravel(y)
    X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    # print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps)

    # print("Computing regularization path using the positive lasso...")
    alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
        X, y, eps=eps, positive=True
    )

    plt.figure(1)
    colors = cycle(["b", "r", "g", "c", "k"])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
    for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
        l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle="--", c=c)

    plt.xlabel("-Log(alpha)")
    plt.ylabel("coefficients")
    plt.title("Lasso and positive Lasso")
    plt.legend((l1[-1], l2[-1]), ("Lasso", "positive Lasso"), loc="lower left")
    plt.axis("tight")
    plt.show()


"""
Main ML loop
"""
# premi??re version
import matplotlib.pyplot as plt
def ml_main(XY, estimator_class):
    X_train, X_test, Y_train, Y_test = XY

    for c in Y_train.columns:
        print_subtitle(f'Estimation of {c}')

        y_train = Y_train.loc[:, c]
        y_test = Y_test.loc[:, c]

        reg = estimator_class().fit(X_train, y_train)
        print(bold('score'), ':', reg.score(X_train, y_train))
        # print(bold('coefs'), ':', reg.coef_)
        print(bold('intercept'), ':', reg.intercept_)

        y_pred = reg.predict(X_test)

        # 5. performance et visualisation des r??sultats
        #print_accuracy(y_pred, y_test) # ici, sur de la r??gression,
        # c'est d??bile : faire une fonction cf. cours.
        print_perf_measures(y_test, y_pred)

        plt.scatter(y_test, y_pred, s=1, color='coral')
        plt.show()

        # x 6 int??gration de la matrice de confusion
        # (ben non, ce n'est pas adapt?? : c'est pour la classif.)


def ml_main_2_ridge(XY):
    X_train, X_test, Y_train, Y_test = XY

    for c in Y_train.columns:
        print_subtitle(f'Ridge estimation of {c}\n')

        y_train = Y_train.loc[:, c]
        y_test = Y_test.loc[:, c]

        baseline_err = get_baseline_err(X_train, y_train, X_test, y_test)
        alpha = best_ridge_alpha(X_train, X_test, y_train, y_test, baseline_err)

        reg = linear_model.Ridge(alpha=alpha).fit(X_train, y_train)

        print(bold('score'), ':', reg.score(X_train, y_train))
        # print(bold('coefs'), ':', reg.coef_)
        print(bold('intercept'), ':', reg.intercept_)

        y_pred = reg.predict(X_test)

        # 5. performance et visualisation des r??sultats
        #print_accuracy(y_pred, y_test) # ici, sur de la r??gression,
        # c'est d??bile : faire une fonction cf. cours.
        print_perf_measures(y_test, y_pred)

        plt.scatter(y_test, y_pred, s=1, color='coral')
        plt.show()

        # x 6 int??gration de la matrice de confusion
        # (ben non, ce n'est pas adapt?? : c'est pour la classif.)


def ml_main_2_lasso(XY):
    X_train, X_test, Y_train, Y_test = XY

    for c in Y_train.columns:
        print_subtitle(f'Lasso estimation of {c}\n')

        y_train = Y_train.loc[:, c]
        y_test = Y_test.loc[:, c]

        baseline_err = get_baseline_err(X_train, y_train, X_test, y_test)

        alpha = best_lasso_alpha(X_train, y_train) # X_test, y_train, y_test, baseline_err)

        reg = linear_model.Lasso(alpha=alpha).fit(X_train, y_train)
        print(bold('score'), ':', reg.score(X_train, y_train))
        # print(bold('coefs'), ':', reg.coef_)
        print(bold('intercept'), ':', reg.intercept_)

        y_pred = reg.predict(X_test)

        # 5. performance et visualisation des r??sultats
        #print_accuracy(y_pred, y_test) # ici, sur de la r??gression,
        # c'est d??bile : faire une fonction cf. cours.
        print_perf_measures(y_test, y_pred)

        plt.scatter(y_test, y_pred, s=1, color='coral')
        plt.show()

        # x 6 int??gration de la matrice de confusion
        # (ben non, ce n'est pas adapt?? : c'est pour la classif.)

def ml_main_3_ridge_scored(XY):
    X_train, X_test, Y_train, Y_test = XY

    scores = empty_scores()

    for c in Y_train.columns:
        print_subtitle(f'Ridge estimation of {c}\n')

        y_train = Y_train.loc[:, c]
        y_test = Y_test.loc[:, c]

        baseline_err = get_baseline_err(X_train, y_train, X_test, y_test)

        alpha = best_ridge_alpha(X_train, X_test, y_train, y_test, baseline_err)

        reg = linear_model.Ridge(alpha=alpha).fit(X_train, y_train)
        print(bold('score'), ':', reg.score(X_train, y_train))
        # print(bold('coefs'), ':', reg.coef_)
        print(bold('intercept'), ':', reg.intercept_)

        y_pred = reg.predict(X_test)

        # 5. performance et visualisation des r??sultats
        #print_accuracy(y_pred, y_test) # ici, sur de la r??gression,
        # c'est d??bile : faire une fonction cf. cours.
        print_perf_measures(y_test, y_pred)

        plt.scatter(y_test, y_pred, s=1, color='coral')
        plt.show()

        score = [
            pd.Timestamp.now(),
            linear_model.Ridge,
            {'alpha': alpha},
            'dataset',
            c,
            'dummy',
            reg.score(X_train, y_train),
            reg.score(X_test, y_test),
            metrics.r2_score(y_test, y_pred),
            '_dummy_score'
        ]
        append_score(scores, score)
    
        # x 6 int??gration de la matrice de confusion
        # (ben non, ce n'est pas adapt?? : c'est pour la classif.)
    return scores


def ml_main_6_scored(dataset, model, param_grid, cv=10, verbose=False):
    X_train, X_test, Y_train, Y_test = dataset._XY

    targets_scores = empty_scores()

    targets_best_params = dict()
    targets_best_score = dict()
    targets_cv_results = dict()

    for c in Y_train.columns:
        if verbose:
            print(f'{model.__name__} estimation of {c} ??? ', end='')
        
        t = time.time()

        y_train = Y_train.loc[:, c]
        y_test = Y_test.loc[:, c]
        Xy = X_train, X_test, y_train, y_test

        baseline_err = get_baseline_err(X_train, y_train, X_test, y_test)

        # best_params = get_best_params(Xy, model, param_grid, baseline_err, cv, verbose)
        #gs.best_estimator_, gs.best_params_, gs.best_score_, gs.best_index_, gs.scorer_, cv_res
        _, best_params, best_score, _, _, cv_results = \
            get_best_params(Xy, model, param_grid, baseline_err, cv, verbose)

        best_est = model(**best_params).fit(X_train, y_train)
        y_pred = best_est.predict(X_test)

        t = time.time() - t
        targets_best_params[c] = best_params
        targets_best_score[c] = best_score
        targets_cv_results[c] = cv_results

        if verbose:
            print(bold('score'), ':', best_score) #est.score(X_train, y_train))
            # print(bold('coefs'), ':', reg.coef_)
            print(bold('intercept'), ':', best_est.intercept_)

        
        if verbose:
            print('time :', round(t, 2), 's')
            print_perf_measures(y_test, y_pred)
            plt.scatter(y_test, y_pred, s=1, color='coral')
            plt.show()

        params = {'split.random_state': dataset.random_state, 'split.test_size': round(dataset.test_size, 3)}
        params.update(best_params)

        score = [
            pd.Timestamp.now(),
            dataset.name,
            c,
            model.__name__,
            params,
            'dummy',
            round(t, 3),
            round(best_est.score(X_train, y_train), 3),
            round(best_est.score(X_test, y_test), 3),
            #metrics.r2_score(y_test, y_pred),
            '_dummy_score'
        ]
        
        append_score(targets_scores, score)
    
    return targets_scores, targets_best_params, targets_best_score, targets_cv_results


def log_scores(dataset, target, model, best_est, best_params, t):
    X_train, X_test, Y_train, Y_test = dataset._XY
    y_train = Y_train.loc[:, target]
    y_test = Y_test.loc[:, target]

    params = {'split.random_state': dataset.random_state, 'split.test_size': round(dataset.test_size, 3)}
    params.update(best_params)
    score = [
        pd.Timestamp.now(),
        dataset.name,
        target,
        model.__name__,
        params,
        'dummy',
        round(t, 3),
        round(best_est.score(X_train, y_train), 3),
        round(best_est.score(X_test, y_test), 3),
        #metrics.r2_score(y_test, y_pred),
        '_dummy_score'
    ]
    
    scores = empty_scores()
    append_score(scores, score)
    return scores


def plot_perfs(est, X, y):
    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(est, X, y, cv=10)

    _, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()


from sklearn.model_selection import cross_val_score
#def ml_main_6_scored_one_target(dataset, target, model, param_grid, cv=10, verbose=False, scale=True):
def search_best_params(dataset, target, model, param_grid, cv=10, verbose=False, scale=True):
    X_train, X_test, Y_train, Y_test = dataset._XY if scale else dataset.XY
    y_train = Y_train.loc[:, target]
    y_test = Y_test.loc[:, target]

    if verbose:
        print(f'{model.__name__} estimation of {target} ??? ', end='')

    t = time.time()

    Xy = X_train, X_test, y_train, y_test

    baseline_err = get_baseline_err(X_train, y_train, X_test, y_test)

    # best_params = get_best_params(Xy, model, param_grid, baseline_err, cv, verbose)
    #gs.best_estimator_, gs.best_params_, gs.best_score_, gs.best_index_, gs.scorer_, cv_res
    _, best_params, best_score, _, _, cv_results = \
        get_best_params(Xy, model, param_grid, baseline_err, cv, verbose)
    
    best_est = model(**best_params).fit(X_train, y_train) # ??a c'est pour l'??val, mais pour la suite, refit
    nested_score = cross_val_score(best_est, X=X_train, y=y_train, cv=cv).mean()
    
    if verbose:
        print(bold('score'), ':', best_est.score(X_train, y_train))
        # print(bold('coefs'), ':', reg.coef_)
        print(bold('intercept'), ':', best_est.intercept_)

    y_pred = best_est.predict(X_test)

    t = time.time() - t
    
    if verbose:
        print('time :', round(t, 2), 's')
        print_perf_measures(y_test, y_pred)

        plot_perfs(best_est, X_test, y_test)

        #plt.scatter(y_test, y_pred, s=1, color='coral')
        #plt.show()

    scores = log_scores(dataset, target, model, best_est, best_params, t)
    return scores, best_params, best_score, nested_score, cv_results
