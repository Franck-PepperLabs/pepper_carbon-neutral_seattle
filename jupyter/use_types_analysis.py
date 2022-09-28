import re
import pandas as pd
from pepper_commons import *

# nb : labels cannot be split with native the split function :
# Personal Services (Health/Beauty, Dry Cleaning, etc) → Personal Services (Health/Beauty | Dry Cleaning | etc)
def tricky_split(s):
    r = re.findall(r'\([^)]*\)|[^()]+', str(s))
    for i, t in enumerate(r):
        if t.startswith('('):
            r[i] = t.replace(', ', '_')
    _s = ''.join(r)
    _r = _s.split(', ')
    _u = [t.replace('_', ', ') for t in _r]
    return _u

def tricky_expand(x):
    return pd.Series(tricky_split(x))


def positions_in(use_series, uses_as_list):
    pos = []
    for id, use in use_series.iteritems():
        try:
            pos += [uses_as_list.loc[id].index(use)]
        except:
            pos += [-1]
    return pd.Series(pos, index=use_series.index)

def check_positions(use_data, use, uses_as_list, label=''):
    use = use[use.notna()]
    use_pos = positions_in(use, uses_as_list)
    not_in_list = use_data.loc[use_pos[use_pos == -1].index]
    print_subtitle(f'Where the {label} use is not in the uses list ({not_in_list.shape[0]} cases)')
    display(not_in_list)
    return use_pos

def not_listed_uses(use, uses_set):
    use = use[use.notna()]
    unique_use = use.unique()
    return list(set(unique_use) - uses_set)


def use_areas(data, uu_index):
    u1 = data.LargestPropertyUseType
    u2 = data.SecondLargestPropertyUseType
    u3 = data.ThirdLargestPropertyUseType
    s1 = data.LargestPropertyUseTypeGFA
    s2 = data.SecondLargestPropertyUseTypeGFA
    s3 = data.ThirdLargestPropertyUseTypeGFA
    areas = pd.Series(index=uu_index, name='area',
        data = [s1[u1 == use].sum() + s2[u2 == use].sum() + s3[u3 == use].sum() for use in uu_index]
    )
    return areas


def unique_uses_table(data):
    uses_list = data.ListOfAllPropertyUseTypes
    uses_list = uses_list.dropna()
    uses_split = uses_list.apply(tricky_expand)
    unique_uses = pd.DataFrame(uses_split.stack().value_counts())
    unique_uses.columns = ['freq']
    unique_uses['area'] = use_areas(data, unique_uses.index)
    total = unique_uses.sum()
    unique_uses[r'%f'] = unique_uses.freq / total.freq
    unique_uses[r'%a'] = unique_uses.area / total.area
    unique_uses = unique_uses.sort_values(by=['area', 'freq'], ascending=False)
    return unique_uses

# généralisation de la précédente : TODO : se passer de la spécifique après tests d'équivalence
# attention, tu vas trop vite mon gars : la précédente travaille avec une colonne de listes d'items !
# notamment utilisée dans seattle_commons pour l'encodage des catégories btype et ptype
def unique_table(data, label):
    g = data[[label, 'PropertyGFATotal']]
    gb = g.groupby(by=label).agg(['count', 'sum'])
    gb.columns = ['freq', 'area']
    totals = gb.sum()
    gb[r'%f'] = gb.freq / totals.freq
    gb[r'%a'] = gb.area / totals.area
    gb = gb.sort_values(by='area', ascending=False)
    return gb


# DEPRECATED => remove all uses and drop it
def use_table(data):
    uu_index = unique_uses_table(data).index
    use_table = pd.DataFrame(index=data.index, columns=uu_index, data=0)
    use_table.insert(0, 'Unknown', 0)
    s = data.PropertyGFATotal
    u1 = data.LargestPropertyUseType
    u2 = data.SecondLargestPropertyUseType
    u3 = data.ThirdLargestPropertyUseType
    s1 = data.LargestPropertyUseTypeGFA
    s2 = data.SecondLargestPropertyUseTypeGFA
    s3 = data.ThirdLargestPropertyUseTypeGFA
    uses_as_list = data.ListOfAllPropertyUseTypes.apply(tricky_split)
    for id, l in uses_as_list.items():
        _0_if_na = lambda s: 0 if str(s) == 'nan' else s
        _1_if_in = lambda u, l: 1 if u[id] in l else 0
        s_i = s[id]
        s1_i, s2_i, s3_i = _0_if_na(s1[id]), _0_if_na(s2[id]), _0_if_na(s3[id])
        n = len(l) - (_1_if_in(u1, l) + _1_if_in(u2, l) + _1_if_in(u3, l))
        if n == 0:
            l += ['Unknown']
            n += 1
        s_others = (s_i - (s1_i + s2_i + s3_i)) / n
        for u in l:
            if u1[id] == u:
                use_table.loc[id, u] = s1_i
            elif u2[id] == u:
                use_table.loc[id, u] = s2_i
            elif u3[id] == u:
                use_table.loc[id, u] = s3_i
            else:
                use_table.loc[id, u] = s_others
    return use_table



# dans cette v2 :
# 1. Unknown n'est plus une colonne ajoutée à la table, mais une Series retournée en seconde sortie
# 2. On complète (les 15 cas d'usages principal et secondaires absents de la liste)
#    et réordonne la liste suivant les surfaces décroissantes
def use_table_2(data, only_table=False):
    uu_index = unique_uses_table(data).index
    use_table = pd.DataFrame(index=data.index, columns=uu_index, data=0)
    s = data.PropertyGFATotal
    
    if not only_table:
        se = data.PropertyGFAParking
        si = data['PropertyGFABuilding(s)']
        s_u = pd.Series(data=0, index=data.index, name='s_u', dtype=int)
        s_diff = pd.Series(data=0, index=data.index, name='s - s_u', dtype=int)

    u1 = data.LargestPropertyUseType
    u2 = data.SecondLargestPropertyUseType
    u3 = data.ThirdLargestPropertyUseType
    s1 = data.LargestPropertyUseTypeGFA
    s2 = data.SecondLargestPropertyUseTypeGFA
    s3 = data.ThirdLargestPropertyUseTypeGFA
    uses_as_list = data.ListOfAllPropertyUseTypes.apply(tricky_split)

    # TODO : tranlate the following in pure vector computing : tricky!
    for id, l in uses_as_list.items():
        # local compact utils
        _is_na = lambda x: str(x) == 'nan'
        _0_if_na = lambda x: 0 if str(x) == 'nan' else x
        _1_if_in = lambda u, l: 1 if u[id] in l else 0
        def _add_if_not_in(_u, l):
            if not (_is_na(_u) or _u in l):
                l = uses_as_list[id] = [_u] + l
        _u1, _u2, _u3 = u1[id], u2[id], u3[id]
        _s, _s1, _s2, _s3 = s[id], _0_if_na(s1[id]), _0_if_na(s2[id]), _0_if_na(s3[id])
        _s_u = int(_s1 + _s2 + _s3)
        _s_diff = _s - _s_u
        if not only_table:
            s_u[id] = _s_u
            s_diff[id] = _s_diff
        # if not already in uses_as_list add u1, u2 and u3 in
        if _is_na(l) or _is_na(l[0]):
            l = uses_as_list[id] = []
        _add_if_not_in(_u3, l)
        _add_if_not_in(_u2, l)
        _add_if_not_in(_u1, l)
        n = len(l) - (_1_if_in(u1, l) + _1_if_in(u2, l) + _1_if_in(u3, l))
        _s_others = 0 if n == 0 else _s_diff / n
        # dispatch each area part in it's target column
        for u in l:
            if u == _u1:
                use_table.loc[id, u] = _s1
            elif u == _u2:
                use_table.loc[id, u] = _s2
            elif u == _u3:
                use_table.loc[id, u] = _s3
            else:
                use_table.loc[id, u] = _s_others

    if only_table:
        return use_table
    else :
        # distinguish between parking and others surfaces
        se_u = pd.Series(data=use_table.Parking, name='se_u', dtype=int)
        se_diff = pd.Series(data=se - se_u, name='se - se_u', dtype=int)
        si_u = pd.Series(data=s_u - se_u, name='si_u', dtype=int)
        si_diff = pd.Series(data=si - si_u, name='si - si_u', dtype=int)
        return use_table, s_u, s_diff, se_u, se_diff, si_u, si_diff