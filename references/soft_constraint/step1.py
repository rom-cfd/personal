#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import axes3d, Axes3D
from pathlib import Path
import os
import time
from helper_functions_v4 import cond_mean
import natsort
from tensorflow import keras
import keras.backend as K
from helper_functions_v3 import do_normalization, do_inverse_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential, load_model
import tensorflow as tf
from keras.layers import GaussianNoise
import math
from keras import regularizers
from keras.layers import Activation
from math import floor
from operator import itemgetter
from joblib import Parallel, delayed
import cantera as ct
import json, datetime


# In[3]:


n_spec=53
n_vectors=n_spec+1
n_diff = n_spec 
n_tran = 6   # avmolwt, lambda, viscosity, cpmix, temp, heat_mass
n_grad = 4


# In[ ]:


data_directory = Path('../../1.make_data_file/preprocessed_data/')

if not data_directory.is_dir():
    raise NotADirectoryError(f"Directory not found: {data_directory}")
        

dns_npz = np.load(data_directory / 'dns_data.npz')

dataTemp = dns_npz["data"]
reacTemp = dns_npz["reac"]
diffTemp = dns_npz["diff"]
tranTemp = dns_npz["tran"]
gradTemp = dns_npz["grad"]



print(f'data shape is {np.shape(dataTemp)}')

data=np.delete(dataTemp, np.s_[n_spec-1], axis=1) # delete N2 mass fractions
print(f"data max value is {np.max(np.max(data,axis=0))}")
print(f"data min value is {np.min(np.min(data,axis=0))}")

reacData=np.delete(reacTemp, np.s_[n_spec-1], axis=1) # delete N2 mass fractions
print(f"reacData max value is {np.max(np.max(reacData,axis=0))}")
print(f"reacData min value is {np.min(np.min(reacData,axis=0))}")

diffData=np.delete(diffTemp, np.s_[n_spec-1], axis=1) # delete N2 mass fractions
print(f"diffData max value is {np.max(np.max(diffData,axis=0))}")
print(f"diffData min value is {np.min(np.min(diffData,axis=0))}")

print(f'N2 remove reactData shape is {np.shape(reacData)}')
print("================================================================")


# In[ ]:


dns_laminar_npz = np.load(data_directory / 'dns_laminar_data.npz')

dataTemp_laminar = dns_laminar_npz["data_laminar"]

print(f'laminar data shape is {np.shape(dataTemp_laminar)}')



data_laminar=np.delete(dataTemp_laminar, np.s_[n_spec-1], axis=1) # delete N2 mass fractions
print(f"data laminar max value is {np.max(np.max(data_laminar,axis=0))}")
print(f"data laminar min value is {np.min(np.min(data_laminar,axis=0))}")

print(f'remove N2 laminar data shape is {np.shape(data_laminar)}')
print("================================================================")


# In[ ]:


spec=['H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'C', 'CH', 'CH2', 'CH2(S)', 'CH3', 'CH4', 'CO', 'CO2', 'HCO', 'CH2O', 'CH2OH', 'CH3O', 'CH3OH', 'C2H', 'C2H2', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'HCCO', 'CH2CO', 'HCCOH', 'N', 'NH', 'NH2', 'NH3', 'NNH', 'NO', 'NO2', 'N2O', 'HNO', 'CN', 'HCN', 'H2CN', 'HCNN', 'HCNO', 'HOCN', 'HNCO', 'NCO', 'AR', 'C3H7', 'C3H8', 'CH2CHO', 'CH3CHO', 'T']


# In[ ]:


print(f'number of species(include T, exclude N2) is {n_spec}')

if n_spec == len(spec):
    print("spec index is proper")
else:
    print("spec index is not proper")

print("================================================================")


# In[8]:


temp_CH4 = [i for i, s in enumerate(spec) if s == "CH4"]
nu_CH4 = temp_CH4[0]

value_to_substract_initial_value = 1e-5
thr_high = np.max(data[:,nu_CH4]) - value_to_substract_initial_value
last_value = 1e-5

initial_fuel_side_idx = np.where(data[:,nu_CH4]>thr_high)[0]
medium_fuel_side_idx = np.where((thr_high>=data[:,nu_CH4]) & (data[:,nu_CH4]>=last_value))[0]
last_fuel_side_idx = np.where(data[:,nu_CH4]<last_value)[0]

initial_fuel_side_idx = initial_fuel_side_idx.tolist()
medium_fuel_side_idx = medium_fuel_side_idx.tolist()
last_fuel_side_idx = last_fuel_side_idx.tolist()

print(f'initial is {len(initial_fuel_side_idx)}')
print(f'medium is {len(medium_fuel_side_idx)}')
print(f'last is {len(last_fuel_side_idx)}')
print(f'total number of data is {len(initial_fuel_side_idx+medium_fuel_side_idx+last_fuel_side_idx)}')


init_fuel_idx = []
medi_fuel_idx = medium_fuel_side_idx
last_fuel_idx = []
for i in range (len(initial_fuel_side_idx)):
    if i%2 != 0:
        init_fuel_idx.append(initial_fuel_side_idx[i])

for i in range (len(last_fuel_side_idx)):
    if i%2 != 0:
        last_fuel_idx.append(last_fuel_side_idx[i])

total_fuel_idx = init_fuel_idx + medi_fuel_idx + last_fuel_idx

print(f'after filtering, initial is {len(init_fuel_idx)}')
print(f'after filtering, medium is {len(medi_fuel_idx)}')
print(f'after filtering, last is {len(last_fuel_idx)}')
print(f'after filtering, total number of data is {len(total_fuel_idx)}')

print('                                                                 ')


# In[9]:


data = data[total_fuel_idx,:]
reacData = reacData[total_fuel_idx,:]
diffData = diffData[total_fuel_idx,:]
tranTemp = tranTemp[total_fuel_idx,:]
gradTemp = gradTemp[total_fuel_idx,:]

print(f'data shape is {np.shape(data)}')
print("================================================================")


# In[10]:


# ===== make input species index =====
species_index = list(range(n_spec))

spec_inp = np.where(np.max(data_laminar, axis=0) > 0.006)[0].tolist()

spec_AR = [i for i, sp in enumerate(spec) if sp == "AR"]
if not spec_AR:
    raise ValueError("This spec list is strange!! spec list must have AR!!")

spec_AR = spec_AR[0]
print(f"AR idx in spec is {spec_AR}")

if spec_AR in spec_inp:
    spec_inp.remove(spec_AR) # remove AR from subset


manual_species = [i for i, s in enumerate(spec) if s == "OH"]
if not manual_species:
    raise ValueError("This spec list is strange!! spec list must have manual species!!")

for ms in manual_species:
    if ms in spec_inp:
        spec_inp.remove(ms)


print(f'number of subset(include T) is {len(spec_inp)}')


# ===== make output species index =====
spec_out = np.delete(species_index, spec_inp, axis=0)
spec_out = spec_out.tolist()

print(f'number of exclude species is {len(spec_out)}')

diff_out = np.delete(spec_inp, -1, axis=0)
diff_out = diff_out.tolist()

specName_inp=itemgetter(*spec_inp)(spec) # extract selected species names as a tuple
specName_out=itemgetter(*diff_out)(spec) # extract selected species names as a tuple
specName_recon=itemgetter(*spec_out)(spec) # extract selected species names as a tuple
print("---------- subset values ----------")
print(specName_inp)
print("---------- subset species ----------")
print(specName_out)
print("---------- recon species ----------")
print(specName_recon)

if not isinstance(specName_inp, tuple):
    specName_inp=(specName_inp,)
if not isinstance(specName_out, tuple):
    specName_out=(specName_out,)
if not isinstance(specName_recon, tuple):
    specName_recon=(specName_recon,)

temp_AR = [i for i, s in enumerate(specName_recon) if s == "AR"]
nu_AR = temp_AR[0]


# In[11]:


data_inp=data[:,spec_inp]
reac_out=reacData[:,diff_out]
data_out=data[:,spec_out]
diff_new=diffData[:,diff_out]
tran=tranTemp
grad=gradTemp
print('input/output data is prepared')
print(f'data inp max is {np.max(data_inp, axis=0)}')
print(f'data inp min is {np.min(data_inp, axis=0)}')
print(f'reac out max is {np.max(reac_out, axis=0)}')
print(f'reac out min is {np.min(reac_out, axis=0)}')
print("================================================================")

prog_var_idx=5
print(f'prog var is {spec[prog_var_idx]}')
prog_var = (data[:,prog_var_idx]) # H2O
prog_var_norm = prog_var/np.max(prog_var, axis=0)

print(f'prog var max value is {np.max(prog_var, axis=0)}')
print(f'prog var norm max value is {np.max(prog_var_norm, axis=0)}')


# In[12]:


del dataTemp
del data
del reacTemp
del reacData
del diffTemp
del diffData
del tranTemp
del gradTemp

# grad input: top-4 species by max over full data_laminar, within subset (T excluded)
_grad_cand = [j for j in spec_inp if j != (n_spec - 1)]
_grad_cand.sort(key=lambda j: np.max(data_laminar[:, j]), reverse=True)
grad4_idx = sorted(_grad_cand[:4])
print(f"grad species (top-4 by max over full data_laminar): {[spec[j] for j in grad4_idx]}")

del data_laminar
del dataTemp_laminar


# In[13]:


# input/output size & ANN layer and node size & number of cluster
n_inp=len(spec_inp)
n_out=len(spec_out)

n_diff=len(diff_out)
n_grad = 4
n_tran = 5
n_flux = 1

print(f'reac ann input/output size is {n_inp}/{n_diff}')
print(f'reconstruct ann input/output size is {n_diff}/{n_out}')
print(f'all species size is {n_inp+n_out}')

n_cl=1

list_spec_hl=[32,32,32,32]
list_diff_hl=[32]
list_tran_hl=list_diff_hl
list_flux_hl=list_diff_hl


# In[14]:


o_ratio=1.05

cl_1 =100

cl_1_inp  = data_inp[np.where((prog_var_norm < o_ratio*cl_1))[0][:], 0:n_inp]  ###  fix!
cl_1_diff  = diff_new[np.where((prog_var_norm < o_ratio*cl_1))[0][:], 0:n_diff]  ###  fix!
cl_1_tran  = tran[np.where((prog_var_norm < o_ratio*cl_1))[0][:], 0:n_tran]  ###  fix!
cl_1_flux  = tran[np.where((prog_var_norm < o_ratio*cl_1))[0][:], -1:]  ###  fix!
cl_1_gradTemp  = grad[np.where((prog_var_norm < o_ratio*cl_1))[0][:], :][:, grad4_idx]  ###  fix!
cl_1_reac_out  = reac_out[np.where((prog_var_norm < o_ratio*cl_1))[0][:], 0:n_diff]  ###  fix!
cl_1_data_out  = data_out[np.where((prog_var_norm < o_ratio*cl_1))[0][:], 0:n_out]  ###  fix!

cl_1_grad = np.concatenate((cl_1_inp, cl_1_gradTemp), axis=1)

print(f'reaction rate output(after prog_var filtering) is {np.shape(cl_1_reac_out)}')

print(f'non value of cl_1_inp is {np.isnan(cl_1_inp).any()}')
print(f'non value of cl_1_reac_out is {np.isnan(cl_1_reac_out).any()}')
print(f'non value of cl_1_diff is {np.isnan(cl_1_diff).any()}')
print(f'non value of cl_1_tran is {np.isnan(cl_1_tran).any()}')
print(f'non value of cl_1_flux is {np.isnan(cl_1_flux).any()}')
print(f'non value of cl_1_grad is {np.isnan(cl_1_grad).any()}')
print(f'non value of cl_1_data_out (reconstruct ANN output) is {np.isnan(cl_1_data_out).any()}')
print("================================================================")


# In[15]:


del data_inp
del reac_out
del diff_new
del tran
del grad
del data_out
del cl_1_gradTemp


# In[16]:


def data_preProcessing(inp,lamb):
    out=np.sign(inp)*(abs(inp))**lamb
    return out

def data_backTransform(inp,lamb):
    out=np.sign(inp)*(abs(inp))**(1/lamb)
    return out


# In[17]:


batch=256
epoch=1000


lamb_cl_1_data = np.zeros(n_inp)
for i in range (0,n_inp):
    lamb_cl_1_data[i]=0.5
lamb_cl_1_data[-1] = 1.0


lamb_cl_1_out = np.zeros(n_diff)
for i in range (0,n_diff):
    lamb_cl_1_out[i] = 0.5

cl_1_data_pre = np.zeros((cl_1_inp.shape[0], cl_1_inp.shape[1]))
cl_1_reac_out_pre = np.zeros((cl_1_reac_out.shape[0], cl_1_reac_out.shape[1]))

for i in range (0,n_inp):
    cl_1_data_pre[:,i] = data_preProcessing(cl_1_inp[:,i],lamb_cl_1_data[i])

for i in range (0,n_diff):
    cl_1_reac_out_pre[:,i] = data_preProcessing(cl_1_reac_out[:,i],lamb_cl_1_out[i])



print(f"batch size is {batch}")
print(f"epoch is {epoch}")
print(f"RR ANN input data pre-processing info is {lamb_cl_1_data}")
print(f"RR ANN output data pre-processing info is {lamb_cl_1_out}")


# In[18]:


del cl_1_inp
del cl_1_reac_out


# In[19]:


(
    cl_1_data_pre_train, cl_1_data_pre_test,
    cl_1_reac_out_pre_train, cl_1_reac_out_pre_test,
    cl_1_grad_train, cl_1_grad_test,
    cl_1_data_out_train, cl_1_data_out_test,
    cl_1_diff_train, cl_1_diff_test,
    cl_1_tran_train, cl_1_tran_test,
    cl_1_flux_train, cl_1_flux_test
) = train_test_split(
    cl_1_data_pre,
    cl_1_reac_out_pre,
    cl_1_grad,
    cl_1_data_out,
    cl_1_diff,
    cl_1_tran,
    cl_1_flux,
    test_size=0.2,    # setting test data size
    shuffle=True,     # setting shuffle on/off
    random_state=42   # setting random seed
)

print('--cl_1_data_pre data train/test split--')
print(f'reac ANN intput train data shape is {np.shape(cl_1_data_pre_train)}')
print(f'reac ANN input test data shape is {np.shape(cl_1_data_pre_test)}')
print("================================================================")


# In[20]:


### data-normalization
input_norm='minmax6'
output_norm='minmax6'

cl_1_redin_train = do_normalization(cl_1_data_pre_train, cl_1_data_pre_train, input_norm)
cl_1_redgrad_train = do_normalization(cl_1_grad_train, cl_1_grad_train, input_norm)
cl_1_redout_train = do_normalization(cl_1_reac_out_pre_train, cl_1_reac_out_pre_train, output_norm)
cl_1_reddiff_train = do_normalization(cl_1_diff_train, cl_1_diff_train, output_norm)
cl_1_redtran_train = do_normalization(cl_1_tran_train, cl_1_tran_train, output_norm)
cl_1_redflux_train = do_normalization(cl_1_flux_train, cl_1_flux_train, output_norm)
cl_1_redreco_train = do_normalization(cl_1_data_out_train, cl_1_data_out_train, output_norm)

cl_1_redin_test = do_normalization(cl_1_data_pre_test, cl_1_data_pre_train, input_norm)
cl_1_redgrad_test = do_normalization(cl_1_grad_test, cl_1_grad_train, input_norm)
cl_1_redout_test = do_normalization(cl_1_reac_out_pre_test, cl_1_reac_out_pre_train, output_norm)
cl_1_reddiff_test = do_normalization(cl_1_diff_test, cl_1_diff_train, output_norm)
cl_1_redtran_test = do_normalization(cl_1_tran_test, cl_1_tran_train, output_norm)
cl_1_redflux_test = do_normalization(cl_1_flux_test, cl_1_flux_train, output_norm)
cl_1_redreco_test = do_normalization(cl_1_data_out_test, cl_1_data_out_train, output_norm)

print(f'reconstruct mass fraction=0 species is {specName_recon[nu_AR]}')

cl_1_redreco_train[:,nu_AR] = 0
cl_1_redreco_test[:,nu_AR] = 0

print(f'cl_1_redin_train max/min = {np.max(cl_1_redin_train)}/{np.min(cl_1_redin_train)}')
print(f'cl_1_redgrad_train max/min = {np.max(cl_1_redgrad_train)}/{np.min(cl_1_redgrad_train)}')
print(f'cl_1_redout_train max/min = {np.max(cl_1_redout_train)}/{np.min(cl_1_redout_train)}')
print(f'cl_1_reddiff_train max/min = {np.max(cl_1_reddiff_train)}/{np.min(cl_1_reddiff_train)}')
print(f'cl_1_redtran_train max/min = {np.max(cl_1_redtran_train)}/{np.min(cl_1_redtran_train)}')
print(f'cl_1_redflux_train max/min = {np.max(cl_1_redflux_train)}/{np.min(cl_1_redflux_train)}')
print(f'cl_1_redreco_train max/min = {np.max(cl_1_redreco_train)}/{np.min(cl_1_redreco_train)}')

print(f'cl_1_redin_test max/min = {np.max(cl_1_redin_test)}/{np.min(cl_1_redin_test)}')
print(f'cl_1_redgrad_test max/min = {np.max(cl_1_redgrad_test)}/{np.min(cl_1_redgrad_test)}')
print(f'cl_1_redout_test max/min = {np.max(cl_1_redout_test)}/{np.min(cl_1_redout_test)}')
print(f'cl_1_reddiff_test max/min = {np.max(cl_1_reddiff_test)}/{np.min(cl_1_reddiff_test)}')
print(f'cl_1_redtran_test max/min = {np.max(cl_1_redtran_test)}/{np.min(cl_1_redtran_test)}')
print(f'cl_1_redflux_test max/min = {np.max(cl_1_redflux_test)}/{np.min(cl_1_redflux_test)}')
print(f'cl_1_redreco_test max/min = {np.max(cl_1_redreco_test)}/{np.min(cl_1_redreco_test)}')
print("================================================================")


# In[2]:


def get_model_species(dimi, nc, hl_config, loss_fn='mse'):

    model = Sequential()
    model.add(Dense(hl_config[0],activation='tanh',input_dim=dimi,kernel_initializer='lecun_normal'))
    model.add(Dense(hl_config[1],activation='tanh',kernel_initializer='lecun_normal'))
    model.add(Dense(hl_config[2],activation='tanh',kernel_initializer='lecun_normal'))
    model.add(Dense(hl_config[3],activation='tanh',kernel_initializer='lecun_normal'))
    model.add(Dense(nc,activation='linear'))
    optimizer = keras.optimizers.Adam(learning_rate=1e-03)
    model.compile(loss=loss_fn, optimizer=optimizer)
    return model

def get_model_single(dimi, nc, hl_config, loss_fn='mse'):
    model = Sequential()
    model.add(Dense(hl_config[0],activation='tanh',input_dim=dimi,kernel_initializer='lecun_normal'))
    model.add(Dense(nc,activation='linear'))
    optimizer = keras.optimizers.Adam(learning_rate=1e-03)
    model.compile(loss=loss_fn, optimizer=optimizer)
    return model


# In[21]:


ann_repo=Path('./ann_models/ann_models_etc/')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-8, min_delta=0.0, verbose=0, mode='min')

Mu=1.0

print(f"Mu: {Mu}")


def train_and_save_model(model_fn, inputs, targets, filename, meta=None):
    model = model_fn()
    model.fit(inputs, targets, epochs=epoch, batch_size=batch, callbacks=[reduce_lr], verbose=0)
    model.save(str(ann_repo / filename))

    if meta is None:
        meta = {}
    meta_path = ann_repo / (filename[:-3] + '.meta.json')
    meta.update({
        'saved_at': datetime.datetime.now().isoformat(timespec="seconds"),
    })
    with open(meta_path, 'w') as f:
        # write dictionary data as json file, indent option is Formatting for Readability
        json.dump(meta, f, indent=2)





def make_ann_soft_loss_minmax6(
    y_min, y_max, A_matrix,
    mu=100.0,
):
    """
    y_min, y_max: shape (n_out,)  (physical-space mins/maxs from TRAIN data)
    A_matrix: shape (n_elem, n_out)  (element mass fraction matrix)
    """

    y_min = tf.constant(y_min, dtype=tf.float32)
    y_max = tf.constant(y_max, dtype=tf.float32)
    A = tf.constant(A_matrix, dtype=tf.float32)
    mu = tf.constant(mu, dtype=tf.float32)

    scale = (y_max - y_min) / 2.0  # shape (n_out,)

    def inv_minmax6(y_norm):
        # y_norm: (batch, n_out) in [-1, 1]
        return (1.0 + y_norm) * scale + y_min

    def ann_soft_loss_fn(y_true_norm, y_pred_norm):
        y_true_norm = tf.cast(y_true_norm, tf.float32)
        y_pred_norm = tf.cast(y_pred_norm, tf.float32)

        # 1) data loss in normalized space
        data_loss = tf.reduce_mean(tf.square(y_pred_norm - y_true_norm))

        # 2) element loss in physical space (inverse-normalized)
        y_true = inv_minmax6(y_true_norm)
        y_pred = inv_minmax6(y_pred_norm)

        # elem = Y @ A^T  => (batch, n_elem)
        elem_true = tf.linalg.matmul(y_true, A, transpose_b=True)
        elem_pred = tf.linalg.matmul(y_pred, A, transpose_b=True)

        element_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(elem_pred - elem_true), axis=1))

        return data_loss + mu * element_loss

    return ann_soft_loss_fn


def build_A_matrix_from_cantera(species_names, gas, elements=None):
    """
    species_names: list[str]  (The output of recon maintains the original species order)
    gas: ct.Solution
    elements: list[str] or None
      - If set to None, it automatically collects all the elements that appear in the species
    return:
      A_matrix: (n_elem, n_species)
      elem_list: elements used (same row order)
    """
    # Determine the list of elements
    if elements is None:
        elem_set = set()
        for sp in species_names:
            elem_set.update(gas.species(sp).composition.keys())
        # The order is fixed (if desired, a priority order such as H, C, O, N, Ar... is also possible)
        elem_list = sorted(elem_set)
    else:
        elem_list = list(elements)

    if 'AR' in elem_list:
        elem_list.remove('AR')
    if 'Ar' in elem_list:
        elem_list.remove('Ar')

    n_elem = len(elem_list)
    n_sp = len(species_names)
    A = np.zeros((n_elem, n_sp), dtype=np.float64)

    # atomic weight (Cantera provides atomic weights indexed by element index)
    # elem_weights[e] = atomic weight
    elem_weights = {e: gas.atomic_weight(e) for e in elem_list}

    for j, sp in enumerate(species_names):
        idx = gas.species_index(sp)
        Wk = gas.molecular_weights[idx]   # Molecular weights
        comp = gas.species(sp).composition  # {'H':2, 'O':1, ...}

        for i, e in enumerate(elem_list):
            a_ek = comp.get(e, 0.0)  # if not exist, 0
            A[i, j] = (a_ek * elem_weights[e]) / Wk

    return A, elem_list


gas = ct.Solution("gri30.yaml")  # With the mechanism file you're using
A_matrix_recon, elem_list = build_A_matrix_from_cantera(specName_recon, gas, elements=None)
print(elem_list, A_matrix_recon.shape)

print("A_matrix:")
print(A_matrix_recon)


def loss_name(loss):
    return getattr(loss, "__name__", str(loss))


y_min = cl_1_data_out_train.min(axis=0)
y_max = cl_1_data_out_train.max(axis=0)


loss_recon = make_ann_soft_loss_minmax6(
    y_min=y_min,
    y_max=y_max,
    A_matrix=A_matrix_recon,   # A, adjusted to the species output of recon
    mu=Mu,
)
loss_other = "mse"





Parallel(n_jobs=4)([
    delayed(train_and_save_model)(
        lambda: get_model_species(n_inp, n_out, list_spec_hl, loss_fn=loss_recon),
        cl_1_redgrad_train[:, 0:n_inp], cl_1_redreco_train,
        'model_spec_cl_1.h5',
        meta={"loss_fn": loss_name(loss_recon), "type": "species_reconstruct", 'A_matrix_shape': list(A_matrix_recon.shape), 'elements': elem_list, 'mu(lambda)': float(Mu)},
    ),
   delayed(train_and_save_model)(
       lambda: get_model_single(n_inp, n_diff, list_diff_hl, loss_fn=loss_other),
       cl_1_redin_train, cl_1_reddiff_train,
       'model_diff_cl_1.h5',
       meta={"loss_fn": loss_name(loss_other), "type": "diff"},
   ),
   delayed(train_and_save_model)(
       lambda: get_model_single(n_inp, n_tran, list_tran_hl, loss_fn=loss_other),
       cl_1_redin_train, cl_1_redtran_train,
       'model_tran_cl_1.h5',
       meta={"loss_fn": loss_name(loss_other), "type": "tran"},
   ),
   delayed(train_and_save_model)(
       lambda: get_model_single(n_inp + n_grad, n_flux, list_flux_hl, loss_fn=loss_other),
       cl_1_redgrad_train, cl_1_redflux_train,
       'model_flux_cl_1.h5',
       meta={"loss_fn": loss_name(loss_other), "type": "flux"},
   )
])


# In[22]:


def root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
    

def calc_r2_scores(y_true, y_pred):
    n_outputs = y_true.shape[1]
    r2_list = []
    for j in range(0, n_outputs):
        r2 = r2_score(y_true[:,j], y_pred[:,j])
        r2_list.append(r2)
    r2_aver = np.mean(r2_list)
    return r2_list, r2_aver
    

def calc_nrmse(y_true, y_pred):
    n_outputs = y_true.shape[1]
    L2_norm_list = []
    for j in range(0, n_outputs):
        L2_norm = root_mean_squared_error(y_true[:,j], y_pred[:,j])/root_mean_squared_error(y_true[:,j], 0.0*y_true[:,j])*100
        L2_norm_list.append(L2_norm)
    L2_norm_aver = np.mean(L2_norm_list)
    return L2_norm_list, L2_norm_aver

mask = []
for i in range(cl_1_redreco_test.shape[1]):
    mask.append(i)

mask.remove(nu_AR)
cl_1_redreco_test = cl_1_redreco_test[:,mask]

error_score_folder = './errorScore/'
error_score_path = Path(error_score_folder)
error_score_path.mkdir(parents=True, exist_ok=True)

for i in range (0,n_cl):
    if mask:
        globals()[f'model_spec_cl_{i+1}']  = load_model(str(ann_repo / f'model_spec_cl_{i+1}.h5'), compile=False)
    globals()[f'model_diff_cl_{i+1}']  = load_model(str(ann_repo / f'model_diff_cl_{i+1}.h5'), compile=False)
    globals()[f'model_tran_cl_{i+1}']  = load_model(str(ann_repo / f'model_tran_cl_{i+1}.h5'), compile=False)
    globals()[f'model_flux_cl_{i+1}']  = load_model(str(ann_repo / f'model_flux_cl_{i+1}.h5'), compile=False)

    if mask:
        spec_pred = globals()[f'model_spec_cl_{i+1}'].predict(cl_1_redgrad_test[:, 0:n_inp])
    diff_pred = globals()[f'model_diff_cl_{i+1}'].predict(cl_1_redin_test)
    tran_pred = globals()[f'model_tran_cl_{i+1}'].predict(cl_1_redin_test)
    flux_pred = globals()[f'model_flux_cl_{i+1}'].predict(cl_1_redgrad_test)

    if mask:
        spec_pred = spec_pred[:,mask]

        r2_list_spec, r2_aver_spec = calc_r2_scores(cl_1_redreco_test, spec_pred)
    r2_list_diff, r2_aver_diff = calc_r2_scores(cl_1_reddiff_test, diff_pred)
    r2_list_tran, r2_aver_tran = calc_r2_scores(cl_1_redtran_test, tran_pred)
    r2_list_flux, r2_aver_flux = calc_r2_scores(cl_1_redflux_test, flux_pred)

    if mask:
        nrmse_list_spec, nrmse_aver_spec = calc_nrmse(cl_1_redreco_test, spec_pred)
    nrmse_list_diff, nrmse_aver_diff = calc_nrmse(cl_1_reddiff_test, diff_pred)
    nrmse_list_tran, nrmse_aver_tran = calc_nrmse(cl_1_redtran_test, tran_pred)
    nrmse_list_flux, nrmse_aver_flux = calc_nrmse(cl_1_redflux_test, flux_pred)

    with open(f"{error_score_folder}errorScore_etc.dat", 'a') as f:
        if mask:
            f.write(f'r2 at spec is {r2_aver_spec:.8f}     NRMSE at spec is {nrmse_aver_spec:.8f}\n')
        f.write(f'r2 at diff is {r2_aver_diff:.8f}     NRMSE at diff is {nrmse_aver_diff:.8f}\n')
        f.write(f'r2 at tran is {r2_aver_tran:.8f}     NRMSE at tran is {nrmse_aver_tran:.8f}\n')
        f.write(f'r2 at flux is {r2_aver_flux:.8f}     NRMSE at flux is {nrmse_aver_flux:.8f}\n')


# In[ ]:




