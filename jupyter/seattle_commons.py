# dataset cleaning
from pepper_commons import *
from pepper_production import drop_class

def clean_dataset(data):
    _data = data.drop(columns=['DataYear', 'City', 'State', 'DefaultData', 'Comments'])
    _data, outliers = drop_class(_data, _data.Outlier.notna(), 'outliers')
    _data, not_compliant = drop_class(_data, _data.ComplianceStatus != 'Compliant', 'not compliant')
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
    return pd.Series(np.log(data.PropertyGFATotal), name='asc')

from use_types_analysis import unique_table
def get_btype_id(data):
    u = unique_table(data, 'BuildingType')
    return pd.Series(data.BuildingType.map(lambda x: u.index.get_loc(x)), name='bid')

def get_ptype_id(data):
    u = unique_table(data, 'PrimaryPropertyType')
    return pd.Series(data.PrimaryPropertyType.map(lambda x: u.index.get_loc(x)), name='pid')