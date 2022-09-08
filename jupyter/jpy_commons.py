# Commons shared between big_off project notebooks

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
        print(path.replace(project_dir, '[project_dir]'), 'created.')
    return path

# input data urls
#data_fields_url = 'https://static.openfoodfacts.org/data/data-fields.txt'
#big_off_url = 'https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv'

data_dir = create_subdir(os.path.join(project_dir, 'data'))
csv_data_dir = create_subdir(os.path.join(data_dir, 'csv'))

"""graph_dir = create_subdir(os.path.join(project_dir, 'grph'))
inputs_dir = create_subdir(os.path.join(data_dir, 'in'))
outputs_dir = create_subdir(os.path.join(data_dir, 'out'))

csv_inputs_dir = create_subdir(os.path.join(inputs_dir, 'csv'))
json_inputs_dir = create_subdir(os.path.join(inputs_dir, 'json'))
xml_inputs_dir = create_subdir(os.path.join(inputs_dir, 'xml'))
txt_inputs_dir = create_subdir(os.path.join(inputs_dir, 'txt'))

csv_outputs_dir = create_subdir(os.path.join(outputs_dir, 'csv'))

big_off_dir = create_subdir(os.path.join(csv_inputs_dir, 'big_off'))
big_off_series_dir = create_subdir(os.path.join(csv_inputs_dir, 'big_off_series'))
big_off_slices_dir = create_subdir(os.path.join(csv_inputs_dir, 'big_off_slices'))

big_off_clean_series_dir = create_subdir(os.path.join(csv_inputs_dir, 'big_off_series'))

csv_outputs_dir = create_subdir(os.path.join(outputs_dir, 'csv'))"""

"""
create_subdir(graph_dir)

create_subdir(csv_inputs_dir)
create_subdir(json_inputs_dir)
create_subdir(txt_inputs_dir)
create_subdir(xml_inputs_dir)

create_subdir(big_off_dir)

create_subdir(csv_outputs_dir)
"""

#data_fields_filename = data_fields_url.split('/')[-1]
#data_fields_path = os.path.join(txt_inputs_dir, data_fields_filename)


# pretty printing
bold = lambda s: '\033[1m' + str(s) + '\033[0m'
italic = lambda s: '\033[3m' + str(s) + '\033[0m'
cyan = lambda s : '\033[36m' + str(s) + '\033[0m'
magenta = lambda s : '\033[35m' + str(s) + '\033[0m'

def print_title(txt):
    print(bold(magenta('\n' + txt.upper())))

def print_subtitle(txt):
    print(bold(cyan('\n' + txt)))

print_title('OFF commons is loaded!')
