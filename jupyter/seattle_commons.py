# dataset cleaning
from pepper_commons import *
from pepper_production import drop_class

def clean_dataset(data, verbose=False):
    _data = data.drop(columns=['DataYear', 'City', 'State', 'DefaultData', 'Comments'])
    _data, outliers = drop_class(_data, _data.Outlier.notna(), 'outliers', verbose)
    _data, not_compliant = drop_class(_data, _data.ComplianceStatus != 'Compliant', 'not compliant', verbose)
    return _data, not_compliant, outliers

import seaborn as sns
def ratio_histo(ratio, num_label='<Numerator>', den_label='<Denominator>', limit=True):
    #main_slope = 3.142857
    #main_line = ok[(ok.ratio / main_slope - 1).abs() < .01]
    m = ratio.median()
    s = ratio.std()
    print(bold('count'), '  :', ratio.count())
    print(bold('median'), ' :', m)
    print(bold('mean'), '   :', ratio.mean())
    print(bold('modes'), '  :', list(ratio.mode()))
    print(bold('std'), '    :', s)
    print(bold('kurt'), '   :', ratio.kurtosis())
    # display(main_slope)
    sns.histplot(data=ratio, kde=True, bins=200, color='green') # , x='ratio'
    plt.title(f'Ratio {num_label} / {den_label}', size=15)
    if limit:
        plt.xlim(m - 4 * s, m + 4 * s)
    plt.show()
    # TODO : faire une petite fonction standard, ça fera moins de code ici

# TODO : Nice-to-have (finitions) : fonction générique, tout ce qui suit réduit à une seule ligne
from itertools import product
def display_outliers(_data, ko, compared_cols, compared_family_other_cols, family_name):
    ko = ko.sort_values(by='ratio', ascending=False)
    id_cols = list(_data.columns[:3])
    ko_infos = _data.loc[ko.index, id_cols + compared_cols + compared_family_other_cols].copy()
    ko_infos.insert(3, 'ratio', ko.ratio)
    ko_infos.columns = pd.MultiIndex.from_tuples(list(product(['Identification'], ['id', 'b_type', 'p_type'])) \
        + list(product(['Ratio'], ko_infos.columns[3:6])) + list(product([family_name], ko_infos.columns[6:])))
    display(ko_infos)


def display_aberrants(_data, ab, compared_cols, compared_family_other_cols, family_name):
    id_cols = list(_data.columns[:3])
    ab_infos = _data.loc[ab.index, id_cols + compared_cols + compared_family_other_cols].copy()
    ab_infos.columns = pd.MultiIndex.from_tuples(list(product(['Identification'], ['id', 'b_type', 'p_type'])) \
        + list(product(['Ratio'], ab_infos.columns[3:5])) + list(product([family_name], ab_infos.columns[5:])))
    display(ab_infos)




""" Fixing of irregularities, return clean Series
"""

def get_id(data):
    # Note - `OSEBuilingID` is set as index and renamed in `id` at loadtime
    return pd.Series(data.index, name='id')     

from use_types_analysis import unique_table
def get_btype_id(data):
    """Custom label encoding of `'BuildingType'`"""
    u = unique_table(data, 'BuildingType')
    return pd.Series(data.BuildingType.map(lambda x: u.index.get_loc(x)), name='bid')

def get_ptype_id(data):
    """Custom label encoding of `'PrimaryPropertyType'`"""
    u = unique_table(data, 'PrimaryPropertyType')
    return pd.Series(data.PrimaryPropertyType.map(lambda x: u.index.get_loc(x)), name='pid')

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def sort_cols_by_freq(ohe_data):
    """Returns `ohe_data`with columns sorted by decreasing frequency."""
    ohe_data.loc['sum'] = ohe_data.sum(axis=0)
    sorted_ohe_data = ohe_data.T.sort_values(by='sum', ascending=False).T
    sorted_ohe_data = sorted_ohe_data.drop(index='sum')
    return sorted_ohe_data

def hot_encode_catvar(data, cat_var, new_name=None, abstractor=None, sort=True):
    cats = data[[cat_var]].copy()
    if new_name is not None:
        cat_var = new_name
        cats.columns = [cat_var]
    if abstractor is not None:
        cats[cat_var] = cats[cat_var].apply(abstractor)
    ohe = OneHotEncoder(handle_unknown='ignore').fit(cats)
    ohe_data = pd.DataFrame.sparse.from_spmatrix(
        ohe.transform(cats),
        index=cats.index,
        columns=ohe.get_feature_names_out()
    ).sparse.to_dense()    # to avoid many warnings
    return sort_cols_by_freq(ohe_data) if sort else ohe_data

def abstract_btype(x):
    """Maps btypes to the three more abstract classes
    `'Multifamily'`, `'NonResidential'` and `'School'`"""
    if 'Multifamily' in x:
        return 'Multifamily'
    elif 'nonresidential' in x.lower():
        return 'NonResidential'
    else:
        return 'School'

def get_abstract_btype(data):
    return data.BuildingType.apply(abstract_btype)

def hot_encode_btype(data, abstractor=abstract_btype):
    return hot_encode_catvar(
        data,
        'BuildingType',
        new_name='btype',
        abstractor=abstractor
    )

def hot_encode_full_btype(data):
    return hot_encode_catvar(
        data,
        'BuildingType',
        new_name='btype',
        abstractor=None
    )

def hot_encode_ptype(data, abstractor=None):
    return hot_encode_catvar(
        data,
        'PrimaryPropertyType',
        new_name='ptype',
        abstractor=abstractor
    ) 

# administrative position
def repair_ngb(x):
    x = x.upper()
    return 'DELRIDGE' if 'DELRIDGE' in x else x

def get_pos_nbg(data):
    return data.Neighborhood.apply(repair_ngb).rename('pos_nbg')

def hot_encode_pos_nbg(data):
    return hot_encode_catvar(
        data,
        'Neighborhood',
        new_name='pos_nbg',
        abstractor=repair_ngb
    ) 

def get_latitude(data):
    return data.Latitude.rename('lat.')

def get_longitude(data):
    return data.Longitude.rename('long.')

def get_age(data):
    """Returns building's age in years (2016 - `YearBuilt`)"""
    return (2016 - data.YearBuilt).rename('age')

def get_n_levels(data):
    return (1 + data.NumberofFloors).rename('n_l')

import numpy as np
def get_log_n_levels(data):
    return np.log(get_n_levels(data)).rename('log(n_l)')

# Gross Areas

def get_area(data):
    """Returns property's gross floor area (`PropertyGFATotal`)"""
    # TODO : repair, mais il n'est pas encore au point : mise au point à déplacer dans un fichier dédié
    return data.PropertyGFATotal.rename('a')

import numpy as np
def get_log_area(data):
    return np.log(get_area(data)).rename('log(a)')

def get_inner_area(data):
    """Returns building's gross floor area (`PropertyGFATotal`)"""
    # TODO : repair, mais il n'est pas encore au point : mise au point à déplacer dans un fichier dédié
    return data['PropertyGFABuilding(s)'].rename('a_i')

def get_log_inner_area(data):
    return np.log(get_inner_area(data)).rename('log(a_i)')

def get_outer_area(data):
    """Returns building's unbuilt gross floor area (`PropertyGFATotal`)"""
    # TODO : repair, mais il n'est pas encore au point : mise au point à déplacer dans un fichier dédié
    return data.PropertyGFAParking.rename('a_o')

def get_area_scale(data):
    """Return the log scaled series of buildings gross floor areas"""
    return pd.Series(np.log(data.PropertyGFATotal), name='a_sc')  # old name : asc

def get_inner_outer_area_distribution(data):
    a = data.PropertyGFATotal
    a_i = get_inner_area(data)
    a_o = get_outer_area(data)
    return pd.concat([pd.Series(a_i / a, name='_a_i'), pd.Series(a_o / a, name='_a_o')], axis=1)


from use_types_analysis import use_table_2
def get_use_area_distribution(data):
    use_table = use_table_2(data, only_table=True)
    # rename columns s_u_0, s_u_1, ..
    use_table.columns = ['s_u_' + str(k) for k in range(len(use_table.columns))]
    # areas normalization
    use_table = use_table.div(data.PropertyGFATotal, axis=0)
    return use_table


def get_use_area_distribution_2(data):
    use_table = use_table_2(data, only_table=True)
    # rename columns _ua_<use_0>, _ua_<use_1>, ..
    use_table.columns = ['_ua_' + label for label in use_table.columns]
    # areas normalization
    use_table = use_table.div(data.PropertyGFATotal, axis=0)
    return use_table

# ENERGYSTAR
def get_star_score(data):
    table = data[['PrimaryPropertyType', 'ENERGYSTARScore']].copy()
    score_median = table.ENERGYSTARScore.median()
    def countna(x):
        return x.isna().sum()

    ref = table.groupby(by='PrimaryPropertyType').agg(['count', 'mean', 'median', countna])
    
    # filling scores for property groups without score
    # # NB : these group med scores can be found on ENERGYStar repository
    no_score_group = ref[ref[('ENERGYSTARScore', 'count')] == 0]
    no_score_group_list = list(no_score_group.index)
    is_in_no_score_group = table.PrimaryPropertyType.isin(no_score_group_list)
    table.loc[is_in_no_score_group, 'ENERGYSTARScore'] = score_median

    # filling scores for individual properties without score
    is_in_no_score_group = table.PrimaryPropertyType.isin(no_score_group_list)
    has_no_individual_score = ~is_in_no_score_group & table.ENERGYSTARScore.isna()
    missing_scores_groups = table[has_no_individual_score].groupby(by='PrimaryPropertyType').agg(countna)
    no_individal_score_group_list = list(missing_scores_groups.index)

    for gp in no_individal_score_group_list:
        is_na_and_in_gp = table.ENERGYSTARScore.isna() & (table.PrimaryPropertyType == gp)
        table.loc[is_na_and_in_gp, 'ENERGYSTARScore'] = ref.loc[gp, ('ENERGYSTARScore', 'median')]
    return table.ENERGYSTARScore.rename('star_score')


# Helpers

def get_ratio(data, num_f, den_f, label):
    return pd.Series(num_f(data) / den_f(data), name=label)

def get_product(data, f_a, f_b, label):
    return pd.Series(f_a(data) * f_b(data), name=label)

# Energy consumption (volume)

def get_site_energy_use(data):
    return data['SiteEnergyUse(kBtu)'].rename('e')

import numpy as np
def get_log_site_energy_use(data):
    return np.log(get_site_energy_use(data)).rename('log(e)')

def get_site_energy_use_wn(data):
    return data['SiteEnergyUseWN(kBtu)'].rename('e_wn')

# get_source_energy_use_intensity defined below to avoid a circular import
# get_source_energy_use_wn defined below to avoid a circular import

def get_electricity_energy_use(data):
    return data['Electricity(kBtu)'].rename('e_e')

def get_steam_energy_use(data):
    return data['SteamUse(kBtu)'].rename('e_s')

def get_gas_energy_use(data):
    return data['NaturalGas(kBtu)'].rename('e_g')

def get_ghge_emissions(data):
    return (1000 * data.TotalGHGEmissions).rename('g')

def get_log_ghge_emissions(data):
    return np.log(get_ghge_emissions(data)).rename('log(g)')

# Energy consumption (intensity)

def get_site_energy_use_intensity(data):
    return get_ratio(data, get_site_energy_use, get_area, 'ie')

def get_site_energy_use_inner_intensity(data):
    return get_ratio(data, get_site_energy_use, get_inner_area, '_ie')

def get_site_energy_use_wn_intensity(data):
    return get_ratio(data, get_site_energy_use_wn, get_area, 'ie_wn')

def get_site_energy_use_wn_inner_intensity(data):
    return get_ratio(data, get_site_energy_use_wn, get_inner_area, '_ie_wn')

def get_source_energy_use_intensity(data):
    return data['SourceEUI(kBtu/sf)'].rename('ies')

def get_source_energy_use(data):        # no built-in volume data
    get_product(data, get_source_energy_use_intensity, get_area, 'es')

def get_source_energy_use_inner_intensity(data):
    return get_ratio(data, get_source_energy_use, get_inner_area, '_ies')

def get_source_energy_use_wn_intensity(data):
    return data['SourceEUIWN(kBtu/sf)'].rename('ies_wn')

def get_source_energy_use_wn(data):     # no built-in volume data
    get_product(data, get_source_energy_use_wn_intensity, get_area, 'es_wn')

def get_source_energy_use_wn_inner_intensity(data):
    return get_ratio(data, get_source_energy_use_wn, get_inner_area, '_ies_wn')

def get_electricity_intensity(data):
    return get_ratio(data, get_electricity_energy_use, get_area, 'ie_e')

def get_electricity_inner_intensity(data):
    return get_ratio(data, get_electricity_energy_use, get_inner_area, '_ie_e')

def get_gas_intensity(data):
    return get_ratio(data, get_gas_energy_use, get_area, 'ie_g')

def get_gas_inner_intensity(data):
    return get_ratio(data, get_gas_energy_use, get_inner_area, '_ie_g')

def get_steam_intensity(data):
    return get_ratio(data, get_steam_energy_use, get_area, 'ie_s')

def get_steam_inner_intensity(data):
    return get_ratio(data, get_steam_energy_use, get_inner_area, '_ie_s')

def get_ghge_intensity(data):
    return get_ratio(data, get_ghge_emissions, get_area, 'ig')

def get_ghge_inner_intensity(data):
    return get_ratio(data, get_ghge_emissions, get_inner_area, '_ig')

def get_log_ghge_intensity(data):
     return np.log(get_ghge_intensity(data)).rename('log(ig)')


"""
Suppression de mes 18 outliers

Ce sont des observation suspectes (atypiques, mais non nécessairement aberrantes) qui sont récidivistes
(outliers cf. divers aspects de contrôles dans le cadre de l'analyse exploratoire).
Il représentent une trentaine de cas, soit 1 % de la population.

Le tableau GSheet [outliers](https://docs.google.com/spreadsheets/d/1gtTOd-taN9aY8sg4PGY456E2AlsMxi2W_-7kZaCSYlA/edit#gid=1394793908&fvid=1576786478)
permet d'établir cette liste de 18 identifiants.
Dans la version finale, il faudra évidemment la produire programmatiquement à l'aide d'une fonction ad hoc
qui fait la synthèse compacte de mes détections d'outliers établies en analyse exploratoire.

688, 700, 757, 19793, 21524, 23355, 23682, 25431, 25763, 26849, 26973, 49784, 49967, 49968, 49972, 50014, 50082, 50086
"""
def drop_my_outliers(data):
    """Drop 18 identified outliers"""
    my_outliers_index = [
    688, 700, 757, 19793, 21524, 23355, 23682, 25431, 25763,
    26849, 26973, 49784, 49967, 49968, 49972, 50014, 50082, 50086]
    my_outliers = data.loc[my_outliers_index]
    return data.drop(index=my_outliers_index), my_outliers


"""
Séparation du résidentiel et du non résidentiel

Cf. clustering, séparation du résidentiel et non résidentiel sur le critère Multifamily.
On considère comme non résidentiel, en disjonction sur b_type et p_type, la présence du mot clé 'Multifamily'
"""
def is_family(data):
    """Return family boolean index"""
    is_multifamily_building = data.BuildingType.str.contains('Multifamily')
    is_multifamily_use = data.PrimaryPropertyType.str.contains('Multifamily')
    bindex = is_multifamily_building | is_multifamily_use
    bindex.name = 'is_family'
    return bindex

def get_family_buildings(data):
    """Return residential data subset"""
    return data[is_family(data)]

def get_business_buildings(data):
    """Return non-residential data subset"""
    return data[~is_family(data)]


""" By class indexing
"""

import numpy as np
def get_rnr_index_table(data):
    """Returns a dataframe with two cols `'r_id'` and `'nr_id'`
    one of them containing the index and the other NaN"""
    rnr_bindex = is_family(data).reset_index()
    rnr_bindex['r_id'] = rnr_bindex['nr_id'] = np.nan
    rnr_bindex.loc[rnr_bindex.is_family, 'r_id'] = rnr_bindex.id
    rnr_bindex.loc[~rnr_bindex.is_family, 'nr_id'] = rnr_bindex.id
    return rnr_bindex[['r_id', 'nr_id']]

import matplotlib.pyplot as plt
def show_rnr_index_table(data):
    id_split = get_rnr_index_table(data)
    id_split.r_id.plot(label='residential', c='green')
    id_split.nr_id.plot(label='non-residential', c='orange')
    plt.title('OSEBuilingID residential and non-residential indexes')
    plt.legend()
    plt.show()

def show_rnr_indexes(data):
    r_id = get_id(get_family_buildings(data))
    nr_id = get_id(get_business_buildings(data))
    r_id.plot(label='residential', c='green')
    nr_id.plot(label='non-residential', c='orange')
    plt.title('OSEBuilingID residential vs. non-residential indexes')
    plt.legend()
    plt.show()


"""
Composition du.es jeu.x de données pour la modélisation
"""

from pepper_commons import get_data
#from seattle_commons import clean_dataset, drop_my_outliers, get_ml_data
def get_clean_ml_data():
    data = get_data()
    data, not_compliant, outliers = clean_dataset(data)  # drop outliers identified by Seattle
    data, my_outliers = drop_my_outliers(data)           # drop my own outliers (18)
    return get_ml_data(data)

def get_ml_data(data):
    """Return dataset conditionned for machine learning"""
    # les deux types principaux encodés suivant
    # leur fréquence décroissante de surface représentée
    pid = get_ptype_id(data)
    bid = get_btype_id(data)

    x = data.Latitude.rename('x')
    y = data.Longitude.rename('y')
    z = data.NumberofFloors.rename('z')    # hauteur en étages
    t = 2016 - data.YearBuilt.rename('t')  # ancienneté en années

    # la surface totale intervient, mais indirectement, par son ordre de grandeur -> log
    a_scale = get_area_scale(data)  # log(PropertyGFATotal)

    # on adjoint les surfaces extérieure et intérieure relatives,
    # ça pourrait aider à compenser ces incohérences
     # proportions relatives PropertyGFAParking / PropertyGFABuilding(s)
    ei_ad = get_inner_outer_area_distribution(data)

    # pour le moment (23/09) encore avec ses incohérences
    # redistribution des surfaces par usage (67 cas d'usage)
    u_ad = get_use_area_distribution(data)

    # recalcul des intensités qui le peuvent pour éliminer le bruit des erreurs de troncature
    ie = get_site_energy_use_intensity(data)
    ie_wn = get_site_energy_use_wn_intensity(data)
    ies = get_source_energy_use_intensity(data)
    ies_wn = get_source_energy_use_wn_intensity(data)
    ie_g = get_gas_intensity(data)
    ie_s = get_steam_intensity(data)
    ie_e = get_electricity_intensity(data)

    _1000_ih = get_ghge_intensity(data) #  = _data.GHGEmissionsIntensity

    ml_data = pd.concat([
        bid, pid, x, y, z, t,
        a_scale, ei_ad, u_ad,
        ies_wn, ies, ie_wn, ie, ie_e, ie_s, ie_g,
        _1000_ih], axis=1)

    # display(ml_data)
    # display(ml_data.sum())
    return ml_data

_get_getter_config = {
    'bid': get_btype_id,
    'pid': get_ptype_id,
    'btype': hot_encode_btype,
    'Btype': hot_encode_full_btype,
    'ptype': hot_encode_ptype,
    # ...,
    'pos_nbg': hot_encode_ptype,
    # ...,
    'lat.': get_latitude,
    'long.': get_longitude,
    'T': get_age,
    'n_l': get_n_levels,
    'log(n_l)': get_log_n_levels,
    # ...,
    'a': get_area,
    'log(a)': get_log_area,
    'a_i': get_inner_area,
    'log(a_i)': get_log_inner_area,
    'a_o': get_outer_area,
    '_a_dist': get_inner_outer_area_distribution,
    'asc': get_area_scale,
    # 'ei_ad': get_int_ext_area_distribution, # à renommer autrement, pas limpide
    '_ua_dist': get_use_area_distribution_2,
    'star_score': get_star_score,
    'e': get_site_energy_use,
    'log(e)': get_log_site_energy_use,
    'ie': get_site_energy_use_intensity,
    '_ie': get_site_energy_use_inner_intensity,
    'ie_wn': get_site_energy_use_wn_intensity,
    'ies': get_source_energy_use_intensity,
    'ies_wn': get_source_energy_use_wn_intensity,
    'ie_g': get_gas_intensity,
    'ie_s': get_steam_intensity,
    'ie_e': get_electricity_intensity,
    'g': get_ghge_emissions,
    'log(g)': get_log_ghge_emissions,
    'ig': get_ghge_intensity,
    'log(ig)': get_log_ghge_intensity,
}

def get_getter(key):
    return _get_getter_config[key] if key in _get_getter_config else None

def new_get_ml_data(data, sel=['btype', 'ptype', 'ie']):
    parts = [get_getter(k)(data) for k in sel if k in _get_getter_config]
    return pd.concat(parts, axis=1)


_ml_data_configs = {
    "{a : e}": ['a', 'e'],
    "{_a_i, _a_o, a : e}": ['_a_dist', 'a', 'e'],
    "{_a_i, _a_o : ie}": ['_a_dist', 'ie'],
    "{log(a) : log(e)}": ['log(a)', 'log(e)'],
    "{log(a_i) : log(e)}": ['log(a_i)', 'log(e)'],
    "{log(n_l), log(a) : log(e)}": ['log(n_l)', 'log(a)', 'log(e)'],
    "{n_l, log(a) : log(e)}": ['n_l', 'log(a)', 'log(e)'],
    "{T, log(a) : log(e)}": ['T', 'log(a)', 'log(e)'],
    "{n_★, log(a) : log(e)}": ['star_score', 'log(a)', 'log(e)'],
    "{n_★, log(n_l), log(a) : log(e)}": ['star_score', 'T', 'log(n_l)', 'log(a)', 'log(e)'],
    "{t_b, log(a) : log(e)}": ['btype', 'log(a)', 'log(e)'],
    "{t_p, log(a) : log(e)}": ['ptype', 'log(a)', 'log(e)'],
    "{t_b, t_p, log(a) : log(e)}": ['btype', 'ptype', 'log(a)', 'log(e)'],
    "{(_a_u_k)_k, log(a) : log(e)}": ['_ua_dist', 'log(a)', 'log(e)'],
    "{t_p, T, (_a_u_k)_k, n_★, log(n_l), log(a) : log(e)}":
        ['ptype', 'T', '_ua_dist', 'star_score', 'log(n_l)', 'log(a)', 'log(e)'],
    "{t_B, t_p, T, (_a_u_k)_k, n_★, log(n_l), log(a) : log(e)}":
        ['Btype', 'ptype', 'T', '_ua_dist', 'star_score', 'log(n_l)', 'log(a)', 'log(e)'],

    "{a : g}": ['a', 'g'],
    "{_a_i, _a_o, a : g}": ['_a_dist', 'a', 'g'],
    "{_a_i, _a_o : g}": ['_a_dist', 'g'],
    "{log(a) : log(g)}": ['log(a)', 'log(g)'],
    "{log(a_i) : log(g)}": ['log(a_i)', 'log(g)'],
    "{log(n_l), log(a) : log(g)}": ['log(n_l)', 'log(a)', 'log(g)'],
    "{n_l, log(a) : log(g)}": ['n_l', 'log(a)', 'log(g)'],
    "{T, log(a) : log(g)}": ['T', 'log(a)', 'log(g)'],
    "{n_★, log(a) : log(g)}": ['star_score', 'log(a)', 'log(g)'],
    "{n_★, log(n_l), log(a) : log(g)}": ['star_score', 'T', 'log(n_l)', 'log(a)', 'log(g)'],
    "{t_b, log(a) : log(g)}": ['btype', 'log(a)', 'log(g)'],
    "{t_p, log(a) : log(g)}": ['ptype', 'log(a)', 'log(g)'],
    "{t_b, t_p, log(a) : log(g)}": ['btype', 'ptype', 'log(a)', 'log(g)'],
    "{(_a_u_k)_k, log(a) : log(g)}": ['_ua_dist', 'log(a)', 'log(g)'],
    "{t_p, T, (_a_u_k)_k, n_★, log(n_l), log(a) : log(g)}":
        ['ptype', 'T', '_ua_dist', 'star_score', 'log(n_l)', 'log(a)', 'log(g)'],
    "{t_B, t_p, T, (_a_u_k)_k, n_★, log(n_l), log(a) : log(g)}":
        ['Btype', 'ptype', 'T', '_ua_dist', 'star_score', 'log(n_l)', 'log(a)', 'log(g)'],
}

def get_config(config_name):
    return _ml_data_configs[config_name]

def get_ml_data_cfg(data, config_name):
    return new_get_ml_data(data, sel=get_config(config_name))


"""
Partition
"""

def rnr_mapper(x):
    return 'Residential' if 'Multifamily' in x else 'NonResidential'

from pepper_selection import multipartition
from pepper_skl_commons import Dataset
def get_rnr_datasets(data, name, random_state, test_size):
    map = {'BuildingType': rnr_mapper, 'PrimaryPropertyType': rnr_mapper}
    parts, cats = multipartition(data, map)
    return [
        Dataset(get_ml_data(part), f'{name}[{str(cat)}]', random_state, test_size)
        for part, cat in zip(parts, cats)
        if part.shape[0] > 10]


def get_right_rnr_datasets(data, name, random_state, test_size, min_size=14):      # 8 datasets
    map = {'BuildingType': None, 'PrimaryPropertyType': rnr_mapper}
    parts, cats = multipartition(data, map)
    return [
        Dataset(get_ml_data(part), f'{name}[{str(cat)}]', random_state, test_size)
        for part, cat in zip(parts, cats)
        if part.shape[0] > min_size]

def get_left_rnr_datasets(data, name, random_state, test_size, min_size=14):       # 21 datasets
    map = {'BuildingType': rnr_mapper, 'PrimaryPropertyType': None}
    parts, cats = multipartition(data, map)
    return [
        Dataset(get_ml_data(part), f'{name}[{str(cat)}]', random_state, test_size)
        for part, cat in zip(parts, cats)
        if part.shape[0] > min_size]

def get_fine_grained_datasets(data, name, random_state, test_size, min_size=14):   # 20 datasets
    map = {'BuildingType': None, 'PrimaryPropertyType': None}
    parts, cats = multipartition(data, map)
    return [
        Dataset(get_ml_data(part), f'{name}[{str(cat)}]', random_state, test_size)
        for part, cat in zip(parts, cats)
        if part.shape[0] > min_size]

def get_all_parts_datasets(data, name, random_state, test_size, min_size=14):      # 50 datasets
    datasets = [Dataset(get_ml_data(data), f'{name}', random_state, test_size)]
    datasets += get_right_rnr_datasets(data, name, random_state, test_size, min_size)
    datasets += get_left_rnr_datasets(data, name, random_state, test_size, min_size)
    datasets += get_fine_grained_datasets(data, name, random_state, test_size, min_size)
    return datasets


""" Plotting ml_data
"""

import matplotlib.pyplot as plt
import seaborn as sns
def plot_ml_data(ml_data, btype, x, y, title):
    plot = sns.relplot(data=ml_data, x=x, y=y, hue=btype, col=btype) 
    plot.fig.subplots_adjust(top=.88)
    plot.fig.suptitle(title, fontweight="bold")
    plt.show()

"""from seattle_commons import get_abstract_btype, plot_ml_data
btype = get_abstract_btype(data)   # hue labels
plot_ml_data(ml_data, btype=btype,
             x='a', y='e',
             title="a = PropertyGFATotal → e = SiteEnergyUse(kBtu)"
)"""

def plot_ml_data_all(ml_data, btype):
    #config = get_config(config_name)
    #ml_data = get_ml_data_cfg(data, config_name)
    target = ml_data.columns[-1]
    features = ml_data.columns[:-1]
    for feature in features:
        title = f"{feature} → {target}"
        plot_ml_data(ml_data, btype, feature, target, title)


""" Linear regression
"""

def has_negative(ml_data):
    return (ml_data < 0).any(axis=None)

def show_positive_status(ml_data):
    if has_negative(ml_data):
        print('✘ Negative coefficients :')
        display(ml_data[(ml_data < 0).any(axis=1)])
    else:
        print('✔ All coefficients are positive or null')

def has_na(ml_data):
    return (ml_data.isna()).any(axis=None)

def show_na_status(ml_data):
    if has_na(ml_data):
        print('✘ NA coefficients :')
        display(ml_data[(ml_data < 0).any(axis=1)])
    else:
        print('✔ No NA coefficient')

def check_data(ml_data):
    show_positive_status(ml_data)
    show_na_status(ml_data)

def features_target_split(ml_data):
    return ml_data[ml_data.columns[:-1]], ml_data[ml_data.columns[-1:]]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.utils._testing import ignore_warnings
@ignore_warnings
def ols_nnls_competition(X_train, X_test, y_train, y_test):
    reg_nnls = LinearRegression(positive=True)
    y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
    r2_score_nnls = r2_score(y_test, y_pred_nnls)
    print("NNLS R2 score", r2_score_nnls)

    reg_ols = LinearRegression()
    y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
    r2_score_ols = r2_score(y_test, y_pred_ols)
    print(" OLS R2 score", r2_score_ols)

    _, ax = plt.subplots()
    ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth=0, marker=".")

    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    low = max(low_x, low_y)
    high = min(high_x, high_y)
    ax.plot([low, high], [low, high], ls="--", c=".3", alpha=0.5)
    ax.set_xlabel("OLS regression coefficients", fontweight="bold")
    ax.set_ylabel("NNLS regression coefficients", fontweight="bold")

    return reg_ols, reg_nnls

from pepper_commons import bold
def show_ols_nnls_results(reg_ols, reg_nnls):
    print(bold('features'), ':', reg_ols.feature_names_in_)
    print(bold('intercept'), '(ols) :', reg_ols.intercept_)
    print(bold('intercept'), '(nnls) :', reg_nnls.intercept_)
    print(bold('coefficients'), '(ols) :', reg_ols.coef_)
    print(bold('coefficients'), '(nnls) :', reg_nnls.coef_)

from sklearn.metrics import r2_score
import statsmodels.api as sm
def sm_ols(X_train, X_test, y_train, y_test):
    _X_train = sm.add_constant(X_train)
    _X_test = sm.add_constant(X_test)
    mod = sm.OLS(y_train, _X_train)
    res = mod.fit()
    y_verif_sm_ols = res.predict(_X_train)
    y_pred_sm_ols = res.predict(_X_test)
    r2_verif_sm_ols = r2_score(y_train, y_verif_sm_ols)
    print("SM OLS R2 verif", r2_verif_sm_ols)
    r2_score_sm_ols = r2_score(y_test, y_pred_sm_ols)
    print("SM OLS R2 score", r2_score_sm_ols)
    print(res.summary())


from sklearn.model_selection import cross_val_score
from sklearn.utils._testing import ignore_warnings
@ignore_warnings
def show_ols_scores(reg_ols, X, y, cv=3):
    """Cross validated scoring of OLS regressor"""
    cv_scores = cross_val_score(reg_ols, X, y, cv=cv)
    print('cv scores :', cv_scores)
    print('mean scores :', np.mean(cv_scores))
    print('std scores :', np.std(cv_scores))