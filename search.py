import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import argparse
import numpy as np
from pyteomics import mgf, mass
from dataclasses import dataclass, asdict

import tensorflow as tf
import tensorflow.keras as k
from tensorflow_addons.layers import InstanceNormalization

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str,
                    help='query file path', default='query.mgf')
parser.add_argument('--lib_vec', type=str,
                    help='library vectors (generate using encode.py)', default='lib.vec')
parser.add_argument(
    "--lib_decoy",
    type=str,
    help="Decoy library vectors (generate using encode.py)",
    default="lib_decoy.vec",
)
parser.add_argument('--db_vec', type=str,
                    help='database vectors (generate using encode.py)', default='db.vec')
parser.add_argument(
    "--db_decoy",
    type=str,
    help="Decoy database vectors (generate using encode.py)",
    default="db_decoy.vec",
)
parser.add_argument('--output', type=str,
                    help='output file path', default='search_result.tsv')
parser.add_argument('--ppm', type=float,
                    help='Max PPM tolerance for matches', default=20)

parser.add_argument(
    "--mode",
    type=int,
    help="Search mode. 1: spectral library search; 2: Fasta db search; 3: Mixed search",
    default=3,
)
parser.add_argument(
    "--model", type=str, help="Pretained model path", default="encoder.h5"
)
parser.add_argument(
    "--batch_size", type=int, help="number of spectra per step", default=128
)

args = parser.parse_args()

MAX_MZ = 2000

print("Starting reading inputs")

def read_vec(file):
    with open(file, "rb") as f:
        vec = pickle.load(f)
    return vec

lib = read_vec(args.lib_vec)
lib_decoy = read_vec(args.lib_decoy)
db = read_vec(args.db_vec)
db_decoy = read_vec(args.db_decoy)

for sp in db_decoy[0]:
    sp["decoy"] = True

for sp in lib_decoy[0]:
    sp["decoy"] = True

def build_db():
    peps = []

    lib_vec = []

    if not args.mode == 2:  # not fasta db only
        print("# Starting reading spectral library mgf of:", args.lib)

        spectral = read_mgf(args.lib)
        peps += [sp["pep"] for sp in spectral]

        for batch in iterate(spectral, args.batch_size * 128):
            flatten_sps = [spectra2vector(sp) for sp in spectral]
            lib_vec += model.predict(data_seq(flatten_sps, processor, args.batch_size))

    vec_db = []

    if not args.mode == 1:  # not spectral only
        predfull = predfull(args.predfull_model)

        seqs = readfasta(args.fasta)

        if args.mode == 3:  # mixed
            known = set(sp["pep"] for sp in spectral)

            seqs = [seq for seq in seqs if not seq["pep"] in known]

        peps += [seq["pep"] for seq in seqs]
        # db_sps = predict_spectra(fasta)

        for batch in iterate(seqs, args.batch_size * 128):
            predicted_sps = predfull(batch, args.batch_size)
            vec_db += model.predict(data_seq(predicted_sps, processor, args.batch_size))

    print(
        "Finished,", len(peps), "spectra in total,", len(set(peps)), "unique peptides"
    )

    return peps, vec_db + lib_vec


build_db()
