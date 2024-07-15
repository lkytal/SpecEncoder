import argparse
import numpy as np
import numba as nb
from numba import int32, float32, float64, boolean
import math
from pyteomics import mgf, mass

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras as k


class config(dict):
    def __init__(self, *args, **kwargs):
        super(config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def f4(x):
    return "{0:.4f}".format(x)


def asnp(x):
    return np.asarray(x)


def asnp32(x):
    return np.asarray(x, dtype="float32")


def np32(x):
    return np.array(x, dtype="float32")


def clipn(*kw, sigma=4):
    return np.clip(np.random.randn(*kw), -sigma, sigma) / sigma


def fastmass(pep, ion_type, charge, mod=None, cam=True):
    base = mass.fast_mass(pep, ion_type=ion_type, charge=charge)

    if cam:
        base += 57.021 * pep.count("C") / charge

    if not mod is None:
        base += 15.995 * np.sum(mod == 1) / charge
    return base


def iterate(x, bsz):
    while len(x) > bsz:
        yield x[:bsz]
        x = x[bsz:]
    yield x


class data_seq(k.utils.Sequence):
    def __init__(self, sps, processor, batch_size, shuffle=1, xonly=1):
        self.sps = sps
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.xonly = xonly

    def on_epoch_begin(self, ep):
        if ep > 0 and self.shuffle:
            np.random.shuffle(self.sps)

    def __len__(self):
        return math.ceil(len(self.sps) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.sps))

        if self.xonly:
            return (self.processor(self.sps[start_idx:end_idx]),)
        else:
            return self.processor(self.sps[start_idx:end_idx])


def m1(pep, c=1, **kws):
    return fastmass(pep, ion_type="M", charge=c, **kws)


def ppmdiff(sp, pep):
    mass = fastmass(pep, "M", sp["charge"])
    return ((sp["mass"] - mass) / mass) * 1000000


def ppm(m1, m2):
    return ((m1 - m2) / m1) * 1000000


mono = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032029,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.04768,
    "C": 160.03019,
    "L": 113.08406,
    "I": 113.08406,
    "D": 115.02694,
    "Q": 128.05858,
    "K": 128.09496,
    "E": 129.04259,
    "M": 131.04048,
    "H": 137.05891,
    "F": 147.06441,
    "R": 156.10111,
    "Y": 163.06333,
    "N": 114.04293,
    "W": 186.07931,
    "O": 147.03538,
    "Z": 147.0354,  # oxidaed M
}
mono = {k: v for k, v in sorted(mono.items(), key=lambda item: item[1])}

amino_list = list("ACDEFGHIKLMNPQRSTVWYZ")
oh_dim = len(amino_list) + 3  # one_hot dimension

amino2id = {"*": 0, "]": len(amino_list) + 1, "[": len(amino_list) + 2}
for i, a in enumerate(amino_list):
    amino2id[a] = i + 1

id2amino = {0: "*", len(amino_list) + 1: "]", len(amino_list) + 2: "["}
for a in amino_list:
    id2amino[amino2id[a]] = a

mass_list = asnp32([0] + [mono[a] for a in amino_list] + [0, 0])


@nb.njit
def normalize(it, mode):
    if mode == 0:
        return it
    elif mode == 2:
        return np.sqrt(it)

    elif mode == 3:
        return np.sqrt(np.sqrt(it))

    elif mode == 4:
        return np.sqrt(np.sqrt(np.sqrt(it)))

    return it


@nb.njit
def _remove_precursor(v, pre_mz, c, precision, low, r):
    for delta in (0, 1, 2):
        mz = pre_mz + delta / c
        if mz > 0 and mz >= low:
            pc = round((mz - low) / precision)

            if pc - r < len(v):
                v[max(0, pc - r) : min(len(v), pc + r)] = 0
    return None  # force inline


def remove_precursor(v, pre_mz, c, precision, low, r=1):
    return _remove_precursor(v, pre_mz, c, precision, low, r)


@nb.njit
def filterPeaks(v, _max_peaks):
    if _max_peaks <= 0 or len(v) <= _max_peaks:
        return v

    kth = len(v) - _max_peaks
    peak_thres = np.partition(v, kth)[kth]
    v[v < peak_thres] = 0
    return v


@nb.njit
def flat(v, mz, it, pre, low, use_max):
    for i, x in enumerate(mz):
        pos = int(round((x - low) / pre))

        if pos < 0 or pos >= len(v):
            continue

        if use_max:
            v[pos] = max(v[pos], it[i])
        else:
            v[pos] += it[i]

    return v


@nb.njit
def _vectorlize(
    mz, it, mass, c, precision, dim, low, mode, v, kth, th, de, dn, use_max
):
    it /= np.max(it)

    if dn > 0:
        it[it < dn] = 0

    it = normalize(it, mode)  # pre-scale

    if kth > 0:
        it = filterPeaks(it, _max_peaks=kth)

    flat(v, mz, it, precision, low, use_max)

    if de == 1:
        _remove_precursor(v, mass, c, precision, low, r=1)  # inplace, before scale

    v /= np.max(v)  # final scale, de can change max

    return v


def vectorlize(
    mz,
    it,
    mass,
    c,
    precision,
    dim,
    low,
    mode,
    out=None,
    kth=-1,
    th=-1,
    de=1,
    dn=-1,
    use_max=0,
):
    if out is None:
        out = np.zeros(dim, dtype="float32")
    return _vectorlize(
        asnp32(mz),
        np32(it),
        mass,
        c,
        precision,
        dim,
        low,
        mode,
        out,
        kth,
        th,
        de,
        dn,
        use_max,
    )


def decode(seq2d):
    return np.int32([np.argmax(seq2d[i]) for i in range(len(seq2d))])


def topep(seq):
    return "".join(map(lambda n: id2amino[n], seq)).strip("*[]")


def toseq(pep):
    return np.int32([amino2id[c] for c in pep.upper()])


def what(seq2d):
    return topep(decode(seq2d))


def clean(pep):
    return (
        pep.strip("*[]")
        .replace("I", "L")
        .replace("*", "L")
        .replace("[", "A")
        .replace("]", "R")
    )


def iterate(x, bsz):
    while len(x) > bsz:
        yield x[:bsz]
        x = x[bsz:]
    yield x


def asnp(x):
    return np.asarray(x)


def asnp32(x):
    return np.asarray(x, dtype="float32")


def np32(x):
    return np.array(x, dtype="float32")


def zero32(shape):
    return np.zeros(shape, dtype="float32")


def clipn(*kw, sigma=4):
    return np.clip(np.random.randn(*kw), -sigma, sigma) / sigma


class config(dict):
    def __init__(self, *args, **kwargs):
        super(config, self).__init__(*args, **kwargs)
        self.__dict__ = self


class data_seq(k.utils.Sequence):
    def __init__(self, sps, processor, batch_size, shuffle=1, xonly=1, **kws):
        self.sps = sps
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.xonly = xonly
        self.kws = kws

    def on_epoch_begin(self, ep):
        if ep > 0 and self.shuffle:
            np.random.shuffle(self.sps)

    def __len__(self):
        return math.ceil(len(self.sps) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.sps))

        if self.xonly:
            return (self.processor(self.sps[start_idx:end_idx], **self.kws),)
        else:
            return self.processor(self.sps[start_idx:end_idx], **self.kws)


def fastmass(pep, ion_type, charge, nmod=None, mod=None, cam=True):
    base = mass.fast_mass(pep, ion_type=ion_type, charge=charge)

    if cam:
        base += 57.021 * pep.count("C") / charge  # fixed C modification

    return base


def m1(pep, c=1, **kws):
    return fastmass(pep, ion_type="M", charge=c, **kws)


def ppm(m1, m2):
    return ((m1 - m2) / m1) * 1000000


def ppmdiff(sp, pep=None):
    if pep is None:
        pep = sp["pep"]
    mass = fastmass(pep, "M", sp["charge"], mod=sp["mod"], nmod=sp["nmod"])
    return ((sp["mass"] - mass) / mass) * 1000000


# #### flat and vectorlize

# In[8]:


@nb.njit
def normalize(it, mode):
    if mode == 0:
        return it

    elif mode == 2:
        return np.sqrt(it)

    elif mode == 3:
        return np.sqrt(np.sqrt(it))

    elif mode == 4:
        return np.sqrt(np.sqrt(np.sqrt(it)))

    return it


@nb.njit
def remove_precursor(v, pre_mz, c, precision, low, r):
    for delta in (0, 1, 2):
        mz = pre_mz + delta / c
        if mz > 0 and mz >= low:
            pc = round((mz - low) / precision)

            if pc - r < len(v):
                v[max(0, pc - r) : min(len(v), pc + r)] = 0
    return None  # force inline


def kth(v, k):
    return np.partition(v, k)[k]


# In[10]:


@nb.njit
def mz2pos(mzs, pre, low):
    return round(mzs / pre + low)


@nb.njit
def flat(v, mz, it, pre, low, use_max):
    for i, x in enumerate(mz):
        pos = int(round((x - low) / pre))

        if pos < 0 or pos >= len(v):
            continue

        if use_max:
            v[pos] = max(v[pos], it[i])
        else:
            v[pos] += it[i]

    return v


# In[11]:


@nb.njit
def native_vectorlize(
    mz, it, mass, c, precision, dim, low, mode, v, kth, th, de, dn, use_max
):
    it /= np.max(it)

    if dn > 0:
        it[it < dn] = 0

    it = normalize(it, mode)  # pre-scale

    # if kth > 0: it = filterPeaks(it, _max_peaks=kth)

    flat(v, mz, it, precision, low, use_max)

    if de == 1:
        remove_precursor(v, mass, c, precision, low, r=1)  # inplace, before scale

    v /= np.max(v)  # final scale, de can change max

    return v


def vectorlize(
    mz,
    it,
    mass,
    c,
    precision,
    dim,
    low,
    mode,
    out=None,
    kth=-1,
    th=-1,
    de=1,
    dn=-1,
    use_max=0,
):
    if out is None:
        out = np.zeros(dim, dtype="float32")
    return native_vectorlize(
        asnp32(mz),
        np32(it),
        mass,
        c,
        precision,
        dim,
        low,
        mode,
        out,
        kth,
        th,
        de,
        dn,
        use_max,
    )


def read_mgf(fn, count=-1, default_charge=-1):
    data = mgf.read(
        open(fn, "r"),
        convert_arrays=1,
        read_charges=False,
        dtype="float32",
        use_index=False,
    )

    collision_const = {
        1: 1,
        2: 0.9,
        3: 0.85,
        4: 0.8,
        5: 0.75,
        6: 0.75,
        7: 0.75,
        8: 0.75,
    }
    spectra = []

    for sp in data:
        param = sp["params"]

        if not "charge" in param:
            if default_charge != -1:
                c = default_charge
            else:
                raise AttributeError("MGF contains spectra without charge")
        else:
            c = int(str(param["charge"][0])[0])

        if "seq" in param:
            pep = param["seq"].strip()
        elif "title" in param:
            pep = param["title"].strip()
        else:
            pep = ""

        if "pepmass" in param:
            mass = param["pepmass"][0]
        else:
            mass = float(param["parent"])

        try:
            hcd = param["hcd"]
            if hcd[-1] == "%":
                hcd = float(hcd)
            elif hcd[-2:] == "eV":
                hcd = float(hcd[:-2])
                hcd = hcd * 500 * collision_const[c] / mass
            else:
                raise Exception("Invalid eV format!")
        except:
            hcd = 0

        mz = sp["m/z array"]
        it = sp["intensity array"]

        spectra.append(
            {
                "pep": pep,
                "charge": c,
                "type": 3,
                "nmod": 0,
                "mod": np.zeros(len(pep), "int32"),
                "mass": mass,
                "mz": mz,
                "it": it,
                "nce": hcd,
            }
        )

        if count > 0 and len(spectra) >= count:
            break

    return spectra

