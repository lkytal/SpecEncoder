import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import numpy as np
from pyteomics import mgf, mass
from dataclasses import dataclass, asdict

import tensorflow as tf
import tensorflow.keras as k
from tensorflow_addons.layers import InstanceNormalization

from utils import *

def read_mgf(fn, count=-1, default_charge=-1):
    data = mgf.read(open(fn, "r"), convert_arrays=1, read_charges=False,
                        dtype='float32', use_index=False)
        
    collision_const = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
    spectra = []

    for sp in data:
        param = sp['params']

        if not 'charge' in param:
            if default_charge != -1:
                c = default_charge
            else:
                raise AttributeError("MGF contains spectra without charge")
        else:
            c = int(str(param['charge'][0])[0])

        if 'seq' in param:
            pep = param['seq'].strip()
        elif 'title' in param:
            pep = param['title'].strip()
        else:
            pep = ''

        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])

        try:
            hcd = param['hcd']
            if hcd[-1] == '%':
                hcd = float(hcd)
            elif hcd[-2:] == 'eV':
                hcd = float(hcd[:-2])
                hcd = hcd * 500 * collision_const[c] / mass
            else:
                raise Exception("Invalid eV format!")
        except:
            hcd = 0

        mz = sp['m/z array']
        it = sp['intensity array']

        spectra.append({'pep': pep, 'charge': c, 'type': 3, 'nmod': 0, 'mod': np.zeros(len(pep), 'int32'),
                    'mass': mass, 'mz': mz, 'it': it, 'nce': hcd})

        if count > 0 and len(spectra) >= count:
            break

    return spectra

# convert spectra into model input format
def spectra2vector(spectra):
    return tuple([inputs[key] for key in inputs])

parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str,
                    help='query file path', default='query.mgf')
parser.add_argument('--vec', type=str,
                    help='Vector library file path (generate using encode.py)', default='lib.vec')
parser.add_argument('--output', type=str,
                    help='output file path', default='example.tsv')
parser.add_argument('--ppm', type=float,
                    help='Max PPM tolerance for matches', default=20)

args = parser.parse_args()

MAX_MZ = 2000

print("Starting reading input mgf of:", args.input)
query = read_mgf(args.input)

f = open(args.output, 'w+')
f.writelines(['TITLE\tSeq\tScore\n'])

mass_map = {}

f.close()
print('Finished,', i, 'spectra in total')
