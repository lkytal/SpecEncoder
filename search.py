import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import pickle
import argparse
import numpy as np
from pyteomics import mgf, mass
from dataclasses import dataclass, asdict
import numba as nb

import tensorflow as tf
import tensorflow.keras as k
from tensorflow_addons.layers import InstanceNormalization

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str,
                    help='query file path', default='query.pkl')
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
    help="Search mode. 1: spectral library search; 2: Database search; 3: Mixed search",
    default=3,
)
parser.add_argument(
    "--model", type=str, help="Pretained model path", default="encoder.h5"
)
parser.add_argument(
    "--batch_size", type=int, help="number of spectra per step", default=128
)
parser.add_argument(
    "--fdr", type=float, help="FDR cutoff", default=0.01
)

args = parser.parse_args()

MAX_MZ = 2000


def read_vec(file):
    with open(file, "rb") as f:
        vec = pickle.load(f)
    return vec


print("Starting reading query file:", args.query)
query, query_vecs = read_vec(args.query)

def build_db():
    lib_peps = set()
    decoy_peps = set()

    lib_sps = []
    decoy_sps = []

    lib_vecs = []
    decoy_vecs = []

    if not args.mode == 2:  # not database only
        print("# Starting reading spectral library mgf of:", args.lib)

        lib_sps, lib_vecs = read_vec(args.lib_vec)
        decoy_sps, decoy_vecs = read_vec(args.lib_decoy)

        for sp in decoy_sps:
            sp["decoy"] = True

        lib_peps = set([sp["pep"] for sp in lib_sps])
        decoy_peps = set([sp["pep"] for sp in decoy_sps])

    if not args.mode == 1:  # not spectral only
        print("# Starting reading database library mgf of:", args.db_vec)
        db_sps, db_vecs = read_vec(args.db_vec)
        db_decoy, decoy_vecs = read_vec(args.db_decoy)

        for sp in db_decoy:
            sp["decoy"] = True

        for sp, vec in zip(db_sps, db_vecs):
            if sp["pep"] not in lib_peps:
                lib_sps.append(sp)
                lib_vecs.append(vec)

        for sp, vec in zip(db_decoy, decoy_vecs):
            if sp["pep"] not in decoy_peps:
                decoy_sps.append(sp)
                decoy_vecs.append(vec)

    return lib_sps, lib_vecs, decoy_sps, decoy_vecs

print("Building target and decoy database")
lib_sps, lib_vecs, decoy_sps, decoy_vecs = build_db()


@nb.njit
def fcos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(db, db_vecs, query, query_vecs, ppm_thres=10, bsz=512):
    dbc = [sp["charge"] for sp in db]
    dbm = [sp["mass"] for sp in db]

    max_precurosr_mass = math.ceil(np.max(dbm))

    rate = math.floor(1 / ((max_precurosr_mass + 1) / 1000000 * ppm_thres) / 2)

    msdb = {i: [] for i in range(max_precurosr_mass * rate)}
    for i, sp in enumerate(db):
        msdb[round(sp["mass"] * rate)].append(i)

    rst = [None] * len(query)
    scores = [-1] * len(query)

    candis = []

    print("Searching for candidates")
    pb = k.utils.Progbar(len(query))
    for i, sp in enumerate(query):
        pb.add(1)
        candi = []

        c, mass = sp["charge"], sp["mass"]
        mass_tol = mass * ppm_thres / 1000000

        shifts = math.ceil(mass_tol * rate)

        for shift in range(-shifts - 1, shifts + 2):  # search adjacent bins
            bucketid = round(mass * rate) + shift

            if not bucketid in msdb:
                continue

            for idx in msdb[bucketid]:
                if c != dbc[idx]:
                    continue

                if abs(mass - dbm[idx]) > mass_tol:
                    continue

                candi.append(idx)
        candis.append(candi)

    print("Searching within candidates")
    pb = k.utils.Progbar(len(query))
    for i, vec in enumerate(query_vecs):
        pb.add(1)

        for j, idx in enumerate(candis[i]):
            s = fcos(vec, db_vecs[idx])

            if s > scores[i]:
                scores[i] = s
                rst[i] = idx

    return scores, [None if i is None else db[i] for i in rst]


print("Searching against target database")
scores, matches = search(lib_sps, lib_vecs, query, query_vecs, ppm_thres=args.ppm, bsz=args.batch_size)

print("Searching against decoy database")
decoy_scores, decoy_matches = search(decoy_sps, decoy_vecs, query, query_vecs, ppm_thres=args.ppm, bsz=args.batch_size)


def fdr_cutoff_threshold(st, sf, fdr):
    st.sort()
    sf.sort()

    i, j = 0, 0

    th = 1

    # Loop until we've gone through all of A and B
    while i < len(st) - 1 and j < len(sf):
        if sf[j] <= st[i]:
            # Move the pointer in B and update the threshold
            th = sf[j]
            j += 1
        else:
            th = st[i]
            i += 1

        if (len(sf) - j) / (len(st) - i) <= fdr:
            return th

    return th


score_thres = fdr_cutoff_threshold(scores, decoy_scores, args.fdr)
print(f"FDR cutoff threshold at {args.fdr}:", score_thres)

print("Writing search results to:", args.output)
with open(args.output, "w") as f:
    f.write("Index\tTitle\tPeptide\tCharge\tM/z\tScore\tPPM difference\n")
    for i, (sp, score) in enumerate(zip(matches, scores)):
        if sp is None:  # no match
            continue

        if score < score_thres:
            continue

        ppm_diff = (query[i]["mass"] - sp["mass"]) / query[i]["mass"] * 1000000

        if abs(ppm_diff) > args.ppm:
            continue

        f.write(
            f"{i}\t{query[i]['title']}\t{sp['pep']}\t{sp['charge']}\t{sp['mass']}\t{score}\t{ppm_diff}\n"
        )
