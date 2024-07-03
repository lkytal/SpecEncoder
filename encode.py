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

from predfull import predfull

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


parser = argparse.ArgumentParser()
parser.add_argument('--lib', type=str,
                    help='Spectral library file path', default='lib.mgf')
parser.add_argument('--fasta', type=str,
                    help='fasta database file path', default='seq.fasta')
parser.add_argument('--mode', type=int,
                    help='Search mode. 1: spectral library search; 2: Fasta db search; 3: Mixed search', default=3)
parser.add_argument('--output', type=str,
                    help='output embedding file path', default='db.npy')
parser.add_argument('--model', type=str,
                    help='Pretained model path', default='model.h5')
parser.add_argument('--predfull_model', type=str,
                    help='Pretained predfull model path (for predicting spectra)', default='pm.h5')
parser.add_argument('--batch_size', type=int,
                    help='number of spectra per step', default=128)

args = parser.parse_args()

print('Loading model....')
tf.keras.backend.clear_session()
model = k.models.load_model(args.model, compile=0)

def predict_spectra(seqs):
    pass

def build_db():
    peps = []

    lib_vec = []

    if not args.mode == 2: # not fasta db only        
        print("# Starting reading spectral library mgf of:", args.lib)

        spectral = read_mgf(args.lib)
        peps += [sp['pep'] for sp in spectral]

        for batch in iterate(spectral, args.batch_size * 128):
            flatten_sps = [spectra2vector(sp) for sp in spectral]
            lib_vec += model.predict(data_seq(flatten_sps, processor, args.batch_size))


    vec_db = []

    if not args.mode == 1: # not spectral only
        predfull = predfull(args.predfull_model)

        seqs = readfasta(args.fasta)

        if args.mode == 3: # mixed
            known = set(sp['pep'] for sp in spectral)

            seqs = [seq for seq in seqs if not seq['pep'] in known]
    
        peps += [seq['pep'] for seq in seqs]
        # db_sps = predict_spectra(fasta)
            
        for batch in iterate(seqs, args.batch_size * 128):
            predicted_sps = predfull(batch, args.batch_size)
            vec_db += model.predict(data_seq(predicted_sps, processor, args.batch_size))

            
    print('Finished,', len(peps), 'spectra in total,', len(set(peps)), 'unique peptides')

    return peps, vec_db + lib_vec


build_db()

