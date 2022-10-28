
"""def multipart_wmx(data, hue):
    labels = data[hue].unique()
    ids = [data[hue] == label for label in labels]"""

from pepper_commons import *
from pepper_selection import *
import pandas as pd

_test_data = get_data()

def drop_class(data, bindex, target='outliers', verbose=False):
    if verbose:
        print_subtitle(f'Removal of {target}')
    c = data[bindex].copy()
    n = c.shape[0]
    if verbose:
        display(c.head())
    data = data.drop(index=c.index)
    data = data.drop(columns=bindex.name)
    if verbose:
        print(f'⇒ {n} {target} + {bindex.name} column dropped')
    return data, c


is_lab = _test_data.PrimaryPropertyType == 'Laboratory'

infer = {
    'quantif': lambda d: class_selection(d, True),      # sélection d'une classe A d'éléments x
    'premise': lambda d: class_selection(d, is_lab),    # sélection d'une sous classe P de ceux-ci
    'consequ': lambda d: d.Neighborhood.values          # opération sur P pour produire le résultat
}

def predict(data, infer):
    a = data[infer['quantif'](data)].copy()
    p = a[infer['premise'](a)]
    c = infer['consequ'](p)
    return c

def test_predict(data, infer):
    display(predict(data, infer))

    # conso et emission (intensités) médiane, et std pour chaque sous classe d'usages non résidentiels
    is_compliant = _test_data.ComplianceStatus == 'Compliant'
    compliant = _test_data[is_compliant]
    is_multifamily_building = compliant.BuildingType.str.contains('Multifamily')
    is_multifamily_use = compliant.PrimaryPropertyType.str.contains('Multifamily')
    is_campus = compliant.BuildingType == 'Campus'
    is_nonresidential = compliant.BuildingType == 'NonResidential'
    is_residential = (
        (is_multifamily_building | is_multifamily_use)
        & ~(is_campus | is_nonresidential)
    )
    is_nonresidential = ~is_residential
    non_residential = compliant[is_nonresidential]
    print('# compliants :', compliant.shape[0])
    print('# non resid :', non_residential.shape[0])



def conso_emiss_class_analysis(data):
    base = pd.DataFrame(data.PrimaryPropertyType.value_counts())
    print(type(base))

    for use in base.index:

        u = data[data.PrimaryPropertyType == use]

        u_s = u['PropertyGFATotal']
        u_e = u['SiteEUI(kBtu/sf)']
        u_r = u['GHGEmissionsIntensity']

        base.loc[use, 'n'] = u.shape[0]

        sm = base.loc[use, 'sum_s'] = u_s.sum()
        mi = base.loc[use, 'min_s'] = u_s.min()
        mx = base.loc[use, 'max_s'] = u_s.max()
        mn = base.loc[use, 'mean_s'] = round(u_s.mean(), 1)
        md = base.loc[use, 'median_s'] = round(u_s.median(), 1)
        st = base.loc[use, 'std_s'] = round(u_s.std(), 1)
        #base.loc[use, 'sum_s'] = u_s.sum()
        base.loc[use, 'mean_s_'] = round((mn - mn) / st, 1)
        base.loc[use, 'median_s_'] = round((md - mn) / st, 1)
        base.loc[use, 'std_s_'] = round((st - mn) / st, 1)

        base.loc[use, 'sum_e'] = u_e.sum()
        base.loc[use, 'mean_e'] = round(u_e.mean(), 1)
        base.loc[use, 'median_e'] = round(u_e.median(), 1)
        base.loc[use, 'std_e'] = round(u_e.std(), 1)
        
        base.loc[use, 'sum_r'] = u_r.sum()
        base.loc[use, 'mean_r'] = round(u_r.mean(), 1)
        base.loc[use, 'median_r'] = round(u_r.median(), 1)
        base.loc[use, 'std_r'] = round(u_r.std(), 1)
    
    return base

"""analysis = predict(compliant, {
    'quantif': lambda d: class_selection(d, is_nonresidential),  # dans le contexte du non résidentiel
    'premise': lambda d: class_selection(d, True),               # pour tous
    'consequ': lambda d: conso_emiss_class_analysis(d)
})

_analysis = analysis.copy()
total = analysis.loc['Total'] = _analysis.sum()
min_ = analysis.loc['Min'] = _analysis.min()
max_ = analysis.loc['Max'] = _analysis.max()
display(analysis)

analysis['n'] = 100 * analysis['n'] / analysis.loc['Total', 'n']
analysis['sum_s'] = 100 * analysis['sum_s'] / analysis.loc['Total', 'sum_s']

# Feature scaling calés sur les min et max des non résidentiels
min_min = analysis.loc['Min', 'min_s']
max_max = analysis.loc['Max', 'max_s']
analysis['min_s'] = 100 * (analysis['min_s'] - min_min) / (max_max - min_min)
analysis['max_s'] = 100 * (analysis['max_s'] - min_min) / (max_max - min_min)
analysis['mean_s'] = 100 * (analysis['mean_s'] - min_min) / (max_max - min_min)
analysis['median_s'] = 100 * (analysis['median_s'] - min_min) / (max_max - min_min)
analysis = analysis.drop(columns='PrimaryPropertyType')
display(analysis)
"""