from pepper_commons import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('dark_background')
import seaborn as sns
#sns.set_theme(style='white')
sns.set(rc={'figure.figsize': (10, 6)})       # taille par défaut des images


def assert_const_ratio(_data, var_x, var_y, xlim=None, ylim=None, err=.1):
    """Cette fonction valide l'existence d'une relation linéaire et en déduit les outliers"""
    check = _data[[var_x, var_y]].copy()
    check['ratio'] = check[var_y] / check[var_x]
    med_ratio = check.ratio.median()
    is_err = (check.ratio - med_ratio).abs() > err   # erreur si écart > 1 pour 1000
    inconst = check[is_err].copy()
    check_sz = check.shape[0]
    inconst_sz = inconst.shape[0]
    check['const'] = 'consistant'
    check.loc[is_err, 'const'] = 'not consistant'
    
    if xlim is None:
        x = check[var_x]
        xlim = (x.min(), x.max())
    if ylim is None:
        y = check[var_y]
        ylim = (y.min(), y.max())

    if not inconst.empty:
        print(bold('# inconsistancies (ε = 1 / 1000)'), ':', inconst_sz, f'({100 * inconst_sz / check_sz:.3f} %)')
    else:
        print('consistancy : all is consistant!')

    print(bold('median ratio'), ':', round(med_ratio, 3))
    print(bold('mean ratio'), ':', round(check.ratio.mean(), 3))
    print(bold('ratio std dev'), ':', round(check.ratio.std(), 3))    
    print(bold('ratio kurtosis'), ':', round(check.ratio.kurt(), 3))    
    print(bold('ratio skew'), ':', round(check.ratio.skew(), 3))

    # échelles réelles
    hue = 'const' if inconst_sz / check_sz < .1 else 'ratio'
    ax = sns.scatterplot(data=check, x=var_x, y=var_y, hue=hue)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    xlen = xlim[1] - xlim[0]
    ylen = ylim[1] - ylim[0]
    x = np.arange(*xlim, xlen / 100)
    y = x * med_ratio
    ax.plot(x, y, 'r--', alpha=.2)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.show()


    # échelles logarithmiques
    #ax = check.plot.scatter(x=var_a, y=var_b, c='b', s=.1)
    hue = 'const' if inconst_sz / check_sz < .1 else 'ratio'
    ax = sns.scatterplot(data=check, x=var_x, y=var_y, hue=hue)
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.xlim(max(1, xlim[0]), xlim[1])
    plt.ylim(max(1, ylim[0]), ylim[1])
    xlen = xlim[1] - xlim[0]
    ylen = ylim[1] - ylim[0]
    x = np.arange(max(1, xlim[0]), xlim[1], xlen / 100)
    y = x * med_ratio
    ax.plot(x, y, 'r--', alpha=.2)
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.show()

    return check, inconst

def exclude_aberrants(data, subset=None, is_aberrant=lambda s: s.isna()):
    d = data[data.columns if subset is None else subset].copy()
    is_ab = d.apply(is_aberrant).any(axis=1)
    ab = data[is_ab]
    re = data[~is_ab]
    print(bold('# excluded'), ':', ab.shape[0])
    print(bold('# confirmed'), ':', re.shape[0])
    return re, ab


def assert_relation(data, features, target, relation):
    a = data[features + [target]].copy()
    a.columns = list('xyzabcdefghijklmnopq')[:len(features)] + ['t']
    a[relation] = a.eval(relation)
    re, ab = exclude_aberrants(a)
    ok, ko = assert_const_ratio(re, 't', relation)
    return ok, ko, re, ab


def marginal_emp_dist(bivar, left, right, on='left'):
    """associative table of left and right members of the relation"""
    first, second = (left, right) if on == 'left' else (right, left)
    repr = bivar.groupby(by=first).count()
    repr = repr.sort_values(by=second, ascending=False)
    repr['id'] = [i + 1 for i in range(repr.shape[0])]
    total = repr[second].sum()
    repr['%'] = (repr[second] * (100 / total)).round(1)
    repr = repr.reset_index()
    repr = repr.set_index('id')
    repr.columns = ['labels', 'freq', '%']
    return repr

def rel_pivot(bivar, left, right, on='left', c_ord=None):
    p = bivar.copy()
    p['count'] = 1
    _1st, _2nd = (left, right) if on == 'left' else (right, left)
    p = p.groupby(by=[_1st, _2nd]).count()
    p = p.sort_values(by='count', ascending=False)
    p = p.reset_index()
    p = p.set_index(_1st)
    p = p.pivot(columns=_2nd)
    p.columns = p.columns.droplevel(0)
    if c_ord:
        p = p[c_ord]
    p = p.fillna(0).astype(int)
    p['Total'] = p.sum(axis=1)
    p = p.sort_values(by='Total', ascending=False)
    p.loc['Total'] = p.sum()
    display(p.applymap(lambda x: str(x) if x > 0 else ''))
    return p

def couple_analysis(data, index, left, right):
    couple = data.loc[index, [left, right]].copy()

    print_subtitle(f'Left modalities (field \'{left}\')')
    l = marginal_emp_dist(couple, left, right, on='left')
    nl = l.shape[0]
    print(bold('Lexicon of size'), ':', nl)
    display(l)

    print_subtitle(f'Right modalities (field \'{right}\')')
    r = marginal_emp_dist(couple, left, right, on='right')
    nr = r.shape[0]
    print(bold('Lexicon of size'), ':', nr)
    display(r)

    det = nl <= nr    # la longiforme est préférée sur la largiforme

    # on cherche à présent à produire la table pivot des deux
    st_side = '[Right]' if det else '[Left]'
    st_dims = f'({nr}, {nl})' if det else f'({nl}, {nr})'
    st_mx = f'({right}|{left})' if det else f'({left}|{right})'
    print_subtitle(st_side + ' ' +  st_dims + ' ' + st_mx + ' incidence matrix')

    on = 'right' if det else 'left'
    repr = l if det else r
    cols = list(repr.labels.values)
    mx = rel_pivot(couple, left, right, on=on, c_ord=cols)
    
    return l, r, mx


def bipart(data, class_bid):
    a = o, x = data[class_bid], data[~class_bid]
    n = o.shape[0], x.shape[0]
    s = o.PropertyGFATotal.sum(), x.PropertyGFATotal.sum()
    w = None
    e = o['SiteEnergyUse(kBtu)'].sum(), x['SiteEnergyUse(kBtu)'].sum()
    r = o.TotalGHGEmissions.sum(), x.TotalGHGEmissions.sum()
    return a, n, s, w, e, r

def multipart(data, class_bids):   # class_bids est supposé représenter une partition => verif facile en binaire
    a = [data[i].copy() for i in class_bids]
    n = [a_i.shape[0] for a_i in a]
    s = [a_i.PropertyGFATotal.sum() for a_i in a]
    w = None
    e = [a_i['SiteEnergyUse(kBtu)'].sum() for a_i in a]
    r = [a_i.TotalGHGEmissions.sum() for a_i in a]
    return a, n, s, w, e, r

def multipart_wmx(data, hue):
    labels = data[hue].unique()
    ids = [data[hue] == label for label in labels]

    a_, n_, s_, _, e_, r_ =  multipart(data, ids)
    
    W = pd.DataFrame([n_, s_, e_, r_], index=list('nser'), columns=list(labels))
    N, S, E, R = W.sum(axis=1)

    W_ = W.copy()
    W_.loc['n'] /= N
    W_.loc['s'] /= S
    W_.loc['e'] /= E
    W_.loc['r'] /= R
    W['sum'] = W.sum(axis=1)
    W.loc['sum'] = W.sum()
    W_['sum'] = W_.sum(axis=1)
    W_.loc['sum'] = W_.sum()

    display(W.applymap(lambda x: f'{np.round(x, 2):.2f}'))
    display(W_.applymap(lambda x: f'{np.round(x, 2):.2f}'))