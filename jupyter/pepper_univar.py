## Analyse générique (données quali et quanti)
#  `series_infos` analyse une Series et `dataframe_infos` toutes les Series d'un DataFrame.

import pandas as pd
import scipy.stats as sps
from statsmodels import robust

from pepper_commons import *

def series_infos(s, idx):
    precision = 3
    is_numeric = pd.api.types.is_numeric_dtype(s.dtype)
    is_bool = pd.api.types.is_bool_dtype(s.dtype)
    counts = s.value_counts()
    freqs = s.value_counts(normalize=True)
    n = s.size
    n_na = s.isna().sum()
    n_notna = s.notna().sum()
    n_unique = s.nunique()
    return {
        # Identification, groupe et type
        'idx': idx,
        'group': group(idx),
        'subgroup': subgroup(idx),
        'name': s.name,
        'domain': domain(idx),
        'format': '<NYI>',
        'dtype': s.dtype,
        'astype': '<NYI>',
        'unity': '<NYI>',
        'is_numeric': is_numeric,
        
        # Nombre de valeurs absentes, de valeurs uniques, taux de remplissage
        'n_elts': n,
        'hasnans': s.hasnans,
        'n_unique': n_unique,
        'n_notna': n_notna,
        'n_na': n_na,
        'filling_rate': round(n_notna / n, precision),
        'uniqueness': round(n_unique / n_notna, precision) if n_notna else pd.NA,
        
        # Tendance centrale de la variable numérique
        'val_min': s.min() if is_numeric else None,
        'val_max': s.max() if is_numeric else None,
        'val_mode': str([round(m, 2) for m in s.mode().tolist()]) if is_numeric else None,
        'val_mean': round(s.mean(), precision) if is_numeric else None,
        'val_trim_mean_10pc': sps.trim_mean(s.dropna().values, 0.1) if is_numeric else None, # TODO test : pas certain que ce ne soit pas s.values
        'val_med': s.median() if is_numeric else None,
        
        # Distribution de la variable numérique
        'val_std': round(s.std(), precision) if is_numeric else None,
        'val_interq_range': (s.quantile(0.75) - s.quantile(0.25)) if is_numeric and not is_bool else None,
        'val_med_abs_dev': robust.scale.mad(s.dropna()) if is_numeric else None,
        'val_skew': round(s.skew(), precision) if is_numeric else None,
        'val_kurt': round(s.kurtosis(), precision) if is_numeric else None,
        'interval': [s.min(), s.max()] if is_numeric else None,
        
        # Distribution de la variable catégorielle
        'modalities': list(counts.index) if not is_numeric else None,
        'mod_counts': list(counts) if not is_numeric else None,
        'mod_freqs': list(freqs) if not is_numeric else None,
        
        # Dimensions, empreinte mémoire, méta-données de l'implémentation
        'shape': s.shape,
        'ndim': s.ndim,
        'empty': s.empty,
        'size': s.size,
        'nbytes': s.nbytes,
        'memory_usage': s.memory_usage,     # non reporté dans le dictionnaire de données
        'flags': s.flags,
        'array container type': type(s.array),
        'values container type': type(s.values)
    }

def dataframe_infos(df):
    infos = [pd.Series(series_infos(df[c], df.columns.get_loc(c))) for c in df.columns]
    return pd.DataFrame(infos)


def data_report(data, csv_filename):
    # build the dataframe version
    data_on_data = dataframe_infos(data)

    # some reductions before export
    cut_cat_list = lambda x: None if x is None else x[:30] + ['...']
    cut_num_list = lambda x: None if x is None else x[:30] + [sum(x[30:])]
    data_on_data['modalities'] = data_on_data.modalities.apply(cut_cat_list)
    data_on_data['mod_counts'] = data_on_data.mod_counts.apply(cut_num_list)
    data_on_data['mod_freqs'] = data_on_data.mod_freqs.apply(cut_num_list)
    data_on_data.drop(columns=['memory_usage'], inplace=True)

    # save in CSV
    vars_analysis_path = os.path.join(csv_data_dir, csv_filename)
    data_on_data.to_csv(vars_analysis_path, sep=',', encoding='utf-8', index=False)

    return data_on_data

def data_report_to_gsheet(data_on_data, spread, sheet_name):
    def null_convert(s):  # représentation du vide.. par du vide
        return s.apply(lambda x: '' if x is None or str(x) in ['nan', '[]', '[nan, nan]'] else x)
    def fr_convert(s):    # gestion de la locale dans un monde numérique encore dominé par les anglo-saxons
        return s.apply(lambda x: str(x).replace(',', ';').replace('.', ','))
    exported = data_on_data.copy()
    exported = exported.apply(null_convert)
    exported['shape'] = exported['shape'].apply(lambda x: '\'' + str(x))  # seuls les initiés savent pourquoi
    exported.loc[:, 'filling_rate':'mod_freqs'] = exported.loc[:, 'filling_rate':'mod_freqs'].apply(fr_convert)
    spread.df_to_sheet(exported, sheet=sheet_name, index=False, headers=False, start='A7')
    # display(exported.loc[:, 'filling_rate':'mod_freqs'])  # un dernier contrôle visuel


def cats_freqs(data, label):
    c = data[label]
    f = c.value_counts()
    f.plot.bar()   # TODO : faire plus sympa à l'aide de SNS
    plt.show()
    return f


def cats_weighted_freqs(data, label, wlabels, short_wlabels):
    cw = data[[label] + wlabels]
    aggs = {label: 'count'}
    for wlabel in wlabels:
        aggs[wlabel] = 'sum'
    fw = cw.groupby(by=label).agg(aggs)
    fw.columns = ['count'] + short_wlabels
    fw = fw.sort_values(by=fw.columns[1], ascending=False)
    total = fw.sum()
    fw /= total
    fw.plot.bar(figsize=(.5 * fw.shape[0], 5))   # TODO : faire plus sympa à l'aide de SNS
    plt.show()
    return fw