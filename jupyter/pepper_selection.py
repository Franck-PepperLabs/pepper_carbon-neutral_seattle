import json
import numpy as np
import pandas as pd
from pepper_commons import *

# See app examples : ...
def bwop(o):
    """Return the bitwise operation corresponding to the o operator symbol"""
    if o in ['~', '￢']:
        return lambda x: ~x
    if o in ['&', '⋀']:
        return lambda x, y: x & y
    if o in ['|', '⋁']:
        return lambda x, y: x | y
    if o in ['^', '⊕', '⊻']:
        return lambda x, y: x ^ y
    if o in ['⇒']:
        return lambda x, y: ~x | y
    if o in ['⇔', '≣']:
        return lambda x, y: ~(x ^ y)
    return None

# See app examples : ...
def normalize_operands(x):
    """Check `x` type and format and turn it into list of boolean arrays.

    Returns :
    index : the base index, that can be built-in,
    x : the turned x,
    (x_size, x_j_size) : shape dimensions,
    (x_type, x_j_type) : the original types (for reverse translation of eval result)"""
    x_type = type(x)
    print('type :', x_type)
    if not issubclass(x_type, (list, np.ndarray, pd.Series, pd.Index, pd.DataFrame)):
        raise ValueError('x is not one of these : list, np.ndarray, pd.Series, pd.DataFrame')
    # assert : x is an ordered collection of components (a vector)
    
    x_size = len(x) if x_type == list else x.shape[0]
    print('size :', x_size)
    if x_size == 0:
        raise ValueError('x is the empty set')
    # assert : x is not the empty set
    
    x_0 = x[x.column[0]] if x_type == pd.DataFrame else x[0]
    x_j_type = type(x_0)
    is_single = False
    x_j_size = None
    x_ij_type = None
    index = None
    print('dtype :', x_j_type)
    if issubclass(x_j_type, (int, np.int32, np.int64, np.bool_)):
        is_single = True
        x_ij_type = x_j_type
        x_j_type = x_type
        x_j_size = x_size
        x_size = 1
        x_0 = x
        x = [x_0]
        print('single operand')
        # assert : x is the single operand
    elif issubclass(x_j_type, (list, np.ndarray, pd.Series, pd.Index)):
        x_j_size = x_0.size if issubclass(x_j_type, (pd.Series, pd.Index)) else len(x_0)
        x_ij_type = type(x_0.iloc[0]) if issubclass(x_j_type, pd.Series) else type(x_0[0])
        print('vector of operands')
    else:
        raise TypeError('x type unknown :', x_j_type)

    # build the index in any case
    if x_ij_type == np.bool_:
        index = np.arange(x_j_size)
        # nothing else to do, it's already a bool array
    else: # x_ij_type == int (index, positional indices)
        # translate the x vectors into boolean arrays
        # the index is the union of indexes or indices
        index = set(x_0)
        if not is_single:
            for x_i in x[1:]:
                index |= set(x_i)
        index = list(index)
        index.sort()
        x = [[i in x_j for i in index] for x_j in x]
    
    return index, x, (x_size, x_j_size), (x_type, x_j_type)


def eval(f, x):
    """ Return f(x).
    
    n-arize the binary f (n = len(x)) and returns e = f(x)
    x is a list or nd.array of alternatively booleans ([]), indexes (loc[]), indices (iloc[])
    """
    if len(x) == 1:
        return f(x[0])

    e = x[0]
    for v in x[1:]:
        e = f(e, v)
    return e


def eval_bindex(op='&', props=None, data=None):
    """Returns props reduced by application of op to its unique equivalent
    props is a single or a collection (pd.DataFrame, pd.Series, np.ndarray)
    of one type among:
    1. selection dicts;                                      abstract selection by conditions
    2. boolean arrays;                                       concrete selection by boolean array
    3. indexes (ids or names);                               concrete selection by index
    4. list, np.ndarray, or pd.Series of indices (int)       concrete selection by indice
    booleans are used with [], indexes with loc[], indices (iloc[])
    the result format is the one of props elements : abstract dict, boolean ndarray, index
    If neither props or data are passed in, returns None
    If data is None, eval props without reference to data index
    Id props is None, the eval depends on op applied to (id, ~id) with id = data.index == data.index
    """
    f = bwop(op)
    if data is None and props is None:
        return None

    all = data.index == data.index
    if props is None:
        return eval(f, [all, ~all])

    index, x, (x_size, x_j_size), (x_type, x_j_type) = normalize_operands(props)
    if data is None :
        e = eval(f, x)
        # TODO : reverse transl. dependending on x_j_type
        return e
    
    # general case
    e = eval(f, [all] + x)
    # TODO : reverse transl. dependending on x_j_type
    return e


def _or(data=None, props=None):
    return eval_bindex(data, props, op='|')

def _id(data=None, props=None):
    return _or(data, props[0])

def _and(data=None, props=None):
    return eval_bindex(data, props, op='&')

def _not(data=None, props=None):
    return eval_bindex(data, props, op='~')

def _xor(data=None, props=None):
    return eval_bindex(data, props, op='^')

def _nor(data=None, props=None):
    return _not(data, _or(data, props))

def _nand(data=None, props=None):
    return _not(data, _and(data, props))

def _impl(data=None, props=None):
    return _or(data, [_not(data, props[0]), _id(data, props[1])])

def _impl_nativ(data=None, props=None):
    return eval_bindex(data, props, op='⇔')

def _equiv(data=None, props=None):
    return _not(data, _xor(data, props))

def _equiv_nativ(data=None, props=None):
    return eval_bindex(data, props, op='⇒')



def class_selection(data, is_a):
    """Return that boolean index of data elements that are members of class (that conforms to `is_a`)
    if `is_a` is a boolean constant, apply it to all data index
    if `is_a` is a string or a dict, try to interpret it (js based query language design)
    if `is_a` is a callable, call it with data as argument.
    if `is_a` is a boolean index, intersect it with data index
    if `is_a` is a list of boolean indexes, intersect them all with data index
    if `is_a` is a list of strings, try to interpret each of it and make the conjunction of all
    """
    all_true = data.index == data.index
    if isinstance(is_a, bool):
        return all_true if is_a else ~all_true
    elif callable(is_a):
        return is_a(data)
    elif isinstance(is_a, dict) or isinstance(is_a, str):
        if isinstance(is_a, str):
            is_a = json.loads(is_a)
        # les clés k sont les noms de variables
        # et les valeurs v les littéraux (pour commencer) dans le prédicat data[k] == v
        props = [data[k] == v for k, v in is_a.items()]
        return _and(data, props)
    elif isinstance(is_a, np.ndarray) and is_a.dtype == bool:
        return _and(data, [is_a])
    elif isinstance(is_a, pd.Series) and is_a.dtype == bool:
        return _and(data, [is_a.values])
    elif isinstance(is_a, list) and isinstance(is_a[0], np.ndarray) and is_a[0].dtype == bool:
        return _and(data, is_a)
    elif isinstance(is_a, list) and isinstance(is_a[0], pd.Series) and is_a[0].dtype == bool:
        return _and(data, [is_a_i.values for is_a_i in is_a])
    elif isinstance(is_a, list) and (isinstance(is_a[0], dict) or isinstance(is_a[0], str)):
        if isinstance(is_a[0], str):
            is_a = [json.loads(is_a_i) for is_a_i in is_a]
        return _or(None, [
            _and(data, [data[k] == v for k, v in is_a_i.items()])
            for is_a_i in is_a
        ])


"""
Class selectors
"""

equals = lambda d, l, v: d[l] == v
contains = lambda d, l, v: d[l].str.contains(v)

def assertion_selector(label, value, assertion):
    return {
        'label': label,
        'value' : value,
        'assert' : assertion
    }

def not_selector(*args):
    return {'~': [*args]}

def and_selector(*args):
    return {'&': [*args]}

def or_selector(*args):
    return {'|': [*args]}

def display_selector(sel, depth=0):
    """Pretty prints the class selector definition"""
    if len(sel) == 1:
        for k, v in sel.items():
            print(depth * '  ' + k)
            [display_selector(cs, depth + 1) for cs in v]
    else:
        print(depth * '  ' + f"assert[{sel['label']}, {sel['value']}]")    


def instance_selector(data, sel, kind='barray'):
    """Returns the boolean index of class selection"""
    # TODO : on ne fait pas d'optimisation (ce sera une jolie couche supplémentaire !)
    # on se contente pour le moment d'interpréter
    if len(sel) == 1:
        for k, v in sel.items():
            #print('op', k, 'on', v, 'nb sub :', len(v))
            f = bwop(k)
            x = [instance_selector(data, cs, kind) for cs in v]
            return eval(f, x)
    else:
        #print('eval of', class_sel)
        l = sel['label']
        v = sel['value']
        a = sel['assert']
        return a(data, l, v)


"""
Partition
"""

def partition(data, cats_var, mapper=None):  # mapper=None ≣ identité ≣ lambda x: x
    map = data[cats_var]
    if mapper:
        map = map.apply(mapper)
    cats = map.unique()
    parts = [data[map == c] for c in cats]
    return parts, cats

def show_partition(data, cats_var, mapper=None):
    for part, cat in zip(*partition(data, cats_var, mapper)):
        print_subtitle(cat)
        display(part)

import pandas as pd
def multipartition(data, map):  # mapper=None ≣ identité ≣ lambda x: x
    cats_vars = list(map.keys())
    apply_map = lambda s: s.apply(map[s.name]) if map[s.name] else s
    mapped_cats = data[cats_vars].apply(apply_map)
    cats = list(mapped_cats.value_counts(cats_vars).index)
    # TODO : trouver une méthode plus compacte et matricielle de type data[(cols) == (vals)]
    bis = []
    for cc in cats:
        bi = data.index == data.index
        for mpc, c in zip(mapped_cats.items(), cc):
            bi &= mpc[1] == c
        bis += [bi]
    parts = [data[bi] for bi in bis]
    return parts, cats

