__verbose = False

# useful imports
import os
import json
import requests
import glob as gou
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from gspread_pandas import Spread


# memory assets
from sys import getsizeof
KiB = 2**10
MiB = KiB * KiB
GiB = MiB * KiB
TiB = GiB * KiB
def format_iB(n_bytes):
    if n_bytes < KiB:
        return n_bytes, 'iB'
    elif n_bytes < MiB:
        return round(n_bytes / KiB, 3), 'KiB'
    elif n_bytes < GiB:
        return round(n_bytes / MiB, 3), 'MiB'
    elif n_bytes < TiB:
        return round(n_bytes / GiB, 3), 'GiB'
    else:
        return round(n_bytes / TiB), 'TiB'


# project data dirs
project_dir = os.path.abspath('..')

def print_path_info(path):
    print(path.replace(project_dir, '[project_dir]'), 'exists' if os.path.exists(path) else 'doesn\'t exist')

def create_subdir(project_path, rel_path=''):
    path = os.path.join(project_path, rel_path)
    if __verbose:
        print_path_info(path)
    if not os.path.exists(path):
        os.makedirs(path)
        print(bold('✔ ' + path.replace(project_dir, '[project_dir]')), 'created.')
    return path

data_dir = create_subdir(os.path.join(project_dir, 'data'))
csv_data_dir = create_subdir(os.path.join(data_dir, 'csv'))
json_data_dir = create_subdir(os.path.join(data_dir, 'json'))

data_filename = '2016_Building_Energy_Benchmarking.csv'
data_filepath = os.path.join(csv_data_dir, data_filename)


# pretty printing
bold = lambda s: '\033[1m' + str(s) + '\033[0m'
italic = lambda s: '\033[3m' + str(s) + '\033[0m'
cyan = lambda s : '\033[36m' + str(s) + '\033[0m'
magenta = lambda s : '\033[35m' + str(s) + '\033[0m'
red = lambda s : '\033[31m' + str(s) + '\033[0m'
green = lambda s : '\033[32m' + str(s) + '\033[0m'

def print_title(txt):
    print(bold(magenta('\n' + txt.upper())))

def print_subtitle(txt):
    print(bold(cyan('\n' + txt)))

status = lambda s, o: bold(green('✔ ' + o) if s else red('✘ ' + o))
def commented_return(s, o, a, *args): # ='✔'
    print(status(s, o), a)
    return args

# dataset preload
_data = pd.read_csv(data_filepath)
commented_return(True, '_data', 'loaded')
display(_data) if __verbose else print(end='')
get_data = lambda: _data.copy()


# JSON utils
def load_json_file(json_data_dir, json_filename):
    """ Return the content of json_filename as a dict """
    json_filepath = os.path.join(json_data_dir, json_filename)
    with open(json_filepath, 'r', encoding='utf-8') as jsonfile:
        print(json_filename, 'loaded')
        return json.load(jsonfile)

def data_to_json(data, json_filename):
    json_path = os.path.join(json_data_dir, json_filename)
    data_dict = {i: json.loads(row.to_json()) for i, row in data.iterrows()}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


# GSheet utils
def gsheet_to_df(spread, sheetname, start_row=2, header_rows=3, clean_header=True):
    data = spread.sheet_to_df(index=0, sheet=sheetname, start_row=start_row, header_rows=header_rows)
    if clean_header:
        data.columns = data.columns.droplevel([header_rows - 1])
    return data

def data_to_gsheet(data, spread, sheet_name, as_code=None, as_fr_FR=None):
    # utilitaires locales
    esc = lambda s: s.apply(lambda x: '\'' + str(x))   # escaping of digital texts as text codes
    clr = lambda s: s.apply(                           # clear empty cells
        lambda x: '' if x is None or str(x) in ['nan', '[]', '[nan, nan]'] else x)  
    to_fr = lambda s: s.apply(lambda x: str(x)         # convert formats to fr_FR locale
        .replace(',', ';').replace('.', ',')) 
    inter = lambda a, b: list(set(a) & set(b))         # intersection
    one_of_in = lambda a, b: len(inter(a, b)) > 0
    
    # ajustements des données (formats)
    exported = data.copy()                                   # working copy
    exported = exported.apply(clr)                           # clear empty cells
    as_code = inter(as_code, data.columns)
    as_fr_FR = inter(as_fr_FR, data.columns)
    if as_code:
        exported[as_code] = exported[as_code].apply(esc)
    if as_fr_FR:
        exported[as_fr_FR] = exported[as_fr_FR].apply(to_fr)
    
    spread.df_to_sheet(exported, sheet=sheet_name, index=False, headers=False, start='A10')
    # display(exported.loc[:, 'filling_rate':'mod_freqs'])  # un dernier contrôle visuel


# Multi-indexing utils
def _load_struct():
    data = pd.read_json(os.path.join(json_data_dir, 'struct.json'), typ='frame', orient='index')
    # print(bold('✔ struct'), 'loaded')
    return commented_return(True, 'struct', 'loaded', data)
    #return data

_struct = _load_struct()

# get element by id and label
_get_element = lambda id, label: _struct.loc[_struct.id == id, label].values[0]
group = lambda id: _get_element(id, 'group')
subgroup = lambda id: _get_element(id, 'subgroup')
domain = lambda id: _get_element(id, 'domain')
format = lambda id: _get_element(id, 'format')
unity = lambda id: _get_element(id, 'unity')
astype = lambda id: _get_element(id, 'astype')
nan_code = lambda id: _get_element(id, 'nan_code')
nap_code = lambda id: _get_element(id, 'nap_code')

# get columns labels from ancestor
_get_labels = lambda k, v: _struct.name[_struct[k] == v].values
get_group_labels = lambda gp_label: _get_labels('group', gp_label)

def new_multi_index(levels=['group']):
    return pd.MultiIndex.from_frame(_struct[levels + ['name']], names=levels+['var'])

