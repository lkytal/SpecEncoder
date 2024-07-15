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
parser.add_argument('--mgf', type=str,
                    help='Spectral MGF file path', default='db.mgf')
parser.add_argument('--output', type=str,
                    help='output embedding file path', default='db.pkl')
parser.add_argument('--model', type=str,
                    help='Pretained model path', default='encoder.h5')
parser.add_argument('--batch_size', type=int,
                    help='number of spectra per step', default=128)

args = parser.parse_args()

print('Loading model....')
tf.keras.backend.clear_session()
model = k.models.load_model(args.model, compile=0, safe_mode=False)

print("# Starting reading spectral library mgf of:", args.mgf)
spectra = read_mgf(args.mgf)


# hyper parameter
@dataclass(frozen=True)
class hyper:
    lmax: int = 30
    outlen: int = lmax + 2
    m1max: int = 2048
    mz_max: int = 2048
    pre: float = 0.2
    low: float = 0
    vdim: int = int(mz_max / pre)
    dim: int = vdim + 0
    maxc: int = 8
    sp_dim: int = 4

    mode: int = 3
    scale: float = 0.3


# convert spectra into model input format
def input_processor(spectra):
    nums = len(spectra)

    inputs = config(
        {
            "y": np.zeros([nums, hyper.sp_dim, hyper.dim], "float32"),
            "info": np.zeros([nums, 2], "float32"),
            "charge": np.zeros([nums, hyper.maxc], "float32"),
        }
    )

    for i, sp in enumerate(spectra):
        mass, c, mzs, its = sp["mass"], sp["charge"], sp["mz"], sp["it"]
        mzs = mzs / 1.00052

        its = normalize(its, hyper.mode)

        inputs.info[i][0] = mass / hyper.m1max
        inputs.info[i][1] = sp["type"]
        inputs.charge[i][c - 1] = 1

        precursor_index = min(hyper.dim - 1, round((mass * c - c + 1) / hyper.pre))

        vectorlize(
            mzs,
            its,
            mass,
            c,
            hyper.pre,
            hyper.dim,
            hyper.low,
            0,
            out=inputs.y[i][0],
            use_max=1,
        )
        inputs.y[i][1][:precursor_index] = inputs.y[i][0][:precursor_index][
            ::-1
        ]  # reverse it

        vectorlize(
            mzs,
            its,
            mass,
            c,
            hyper.pre,
            hyper.dim,
            hyper.low,
            0,
            out=inputs.y[i][2],
            use_max=0,
        )
        inputs.y[i][3][:precursor_index] = inputs.y[i][2][:precursor_index][
            ::-1
        ]  # reverse mz

    return tuple([inputs[key] for key in inputs])

print("# Starting encoding spectra....")
vectors = []
for batch in iterate(spectra, args.batch_size * 128):
    vectors += list(model.predict(data_seq(batch, input_processor, args.batch_size)))

with open(args.output, 'wb+') as f:
    pickle.dump([spectra, vectors], f)
    f.close()

print("# Done!")
