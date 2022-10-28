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

    
# fixing of irregularities, return clean Series
def get_ghge_intensity(data):
    h = data.TotalGHGEmissions
    s = data.PropertyGFATotal
    return pd.Series((1000 * h) / s, name='1000 ih')

def get_natural_gas_intensity(data):
    e_g = data['NaturalGas(kBtu)']
    s = data.PropertyGFATotal
    return pd.Series(e_g / s, name='ie_g')

def get_steam_intensity(data):
    e_s = data['SteamUse(kBtu)']
    s = data.PropertyGFATotal
    return pd.Series(e_s / s, name='ie_s')

def get_electricity_intensity(data):
    e_e = data['Electricity(kBtu)']
    s = data.PropertyGFATotal
    return pd.Series(e_e / s, name='ie_e')

def get_site_energy_use_intensity(data):
    e = data['SiteEnergyUse(kBtu)']
    s = data.PropertyGFATotal
    return pd.Series(e / s, name='ie')

def get_site_wn_energy_use_intensity(data):
    e = data['SiteEnergyUseWN(kBtu)']
    s = data.PropertyGFATotal
    return pd.Series(e / s, name='ie_wn')

def get_source_energy_use_intensity(data):
    ies = data['SourceEUI(kBtu/sf)']
    return pd.Series(ies, name='ies')

def get_source_wn_energy_use_intensity(data):
    ies_wn = data['SourceEUIWN(kBtu/sf)']
    return pd.Series(ies_wn, name='ies_wn')

from use_types_analysis import use_table_2
def get_use_area_distribution(data):
    use_table = use_table_2(data, only_table=True)
    # rename columns s_u_0, s_u_1, ..
    use_table.columns = ['s_u_' + str(k) for k in range(len(use_table.columns))]
    # areas normalization
    use_table = use_table.div(data.PropertyGFATotal, axis=0)
    return use_table

def get_int_ext_area_distribution(data):
    s = data.PropertyGFATotal
    s_e = data.PropertyGFAParking
    s_i = data['PropertyGFABuilding(s)']
    return pd.concat([pd.Series(s_e / s, name='s_e'), pd.Series(s_i / s, name='s_i')], axis=1)

def get_area_scale(data):
    """Return the log scaled series of buildings gross floor areas"""
    return pd.Series(np.log(data.PropertyGFATotal), name='asc')

from use_types_analysis import unique_table
def get_btype_id(data):
    u = unique_table(data, 'BuildingType')
    return pd.Series(data.BuildingType.map(lambda x: u.index.get_loc(x)), name='bid')

def get_ptype_id(data):
    u = unique_table(data, 'PrimaryPropertyType')
    return pd.Series(data.PrimaryPropertyType.map(lambda x: u.index.get_loc(x)), name='pid')


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
def get_family_buildings(data):
    """Return residential data subset"""
    is_multifamily_building = data.BuildingType.str.contains('Multifamily')
    is_multifamily_use = data.PrimaryPropertyType.str.contains('Multifamily')
    family = is_multifamily_building | is_multifamily_use
    return data[family]

def get_business_buildings(data):
    """Return non-residential data subset"""
    is_multifamily_building = data.BuildingType.str.contains('Multifamily')
    is_multifamily_use = data.PrimaryPropertyType.str.contains('Multifamily')
    business = ~(is_multifamily_building | is_multifamily_use)
    return data[business]


"""
Composition du.es jeu.x de données pour la modélisation
"""
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
    ei_ad = get_int_ext_area_distribution(data)

    # pour le moment (23/09) encore avec ses incohérences
    # redistribution des surfaces par usage (67 cas d'usage)
    u_ad = get_use_area_distribution(data)

    # recalcul des intensités qui le peuvent pour éliminer le bruit des erreurs de troncature
    ie = get_site_energy_use_intensity(data)
    ie_wn = get_site_wn_energy_use_intensity(data)
    ies = get_source_energy_use_intensity(data)
    ies_wn = get_source_wn_energy_use_intensity(data)
    ie_g = get_natural_gas_intensity(data)
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


from pepper_commons import get_data
#from seattle_commons import clean_dataset, drop_my_outliers, get_ml_data
def get_clean_ml_data():
    data = get_data()
    data, not_compliant, outliers = clean_dataset(data)  # drop outliers identified by Seattle
    data, my_outliers = drop_my_outliers(data)           # drop my own outliers (18)
    return get_ml_data(data)


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