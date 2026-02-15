#!/usr/bin/env python
# coding=utf-8

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import re

def mmx(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def to_complex(s):
    return complex(s.replace('i', 'j'))

def csi2pdp(csi_real, csi_imag, apply_fftshift=True):

    csi = torch.complex(csi_real, csi_imag)


    if apply_fftshift:
        csi = torch.fft.fftshift(csi, dim=1)

    cir = torch.fft.ifft(csi, dim=1)


    pdp = torch.abs(cir) ** 2

    return pdp/torch.max(pdp, dim=0)[0]

class PM_Dataset(Dataset):
    def __init__(self, root_path, fold5_path, train_test_list):

        # 路径设置
        self.train_or_test = train_test_list[:-5]
        self.path = root_path
        self.list_path = fold5_path
        self.Traj = os.path.join(self.path, 'Position_Mall')
        self.cir3 = os.path.join(self.path,'CIR_Mall','AP3')
        self.cir4 = os.path.join(self.path,'CIR_Mall','AP4')
        self.cir5 = os.path.join(self.path,'CIR_Mall','AP5')
        self.cir6 = os.path.join(self.path, 'CIR_Mall', 'AP6')
        self.cir7 = os.path.join(self.path, 'CIR_Mall', 'AP7')
        self.cir8 = os.path.join(self.path, 'CIR_Mall', 'AP8')


        # 奇怪的python排序机制
        self.Traj_set = sorted(os.listdir(self.Traj),key=lambda x: int(re.search(r'_(\d+)', x).group(1)))
        self.cir3_set = sorted(os.listdir(self.cir3),key=lambda x: int(re.search(r'_(\d+)\.txt$', x).group(1)))
        self.cir4_set = sorted(os.listdir(self.cir4),key=lambda x: int(re.search(r'_(\d+)\.txt$', x).group(1)))
        self.cir5_set = sorted(os.listdir(self.cir5),key=lambda x: int(re.search(r'_(\d+)\.txt$', x).group(1)))
        self.cir6_set = sorted(os.listdir(self.cir6),key=lambda x: int(re.search(r'_(\d+)\.txt$', x).group(1)))
        self.cir7_set = sorted(os.listdir(self.cir7),key=lambda x: int(re.search(r'_(\d+)\.txt$', x).group(1)))
        self.cir8_set = sorted(os.listdir(self.cir8),key=lambda x: int(re.search(r'_(\d+)\.txt$', x).group(1)))

        self.idxlist = sorted(list(np.loadtxt(os.path.join(fold5_path,train_test_list))))

    def __len__(self):
        return len(self.idxlist)

    def __getitem__(self, idx):
#### CIR ####
        cir3 = np.loadtxt(os.path.join(self.cir3, self.cir3_set[int(self.idxlist[idx]-1)]), delimiter=',', dtype=str)
        cir3 = np.vectorize(lambda x: complex(x.replace('i', 'j')))(cir3)
        cir3 = mmx(np.abs(cir3)[:, 0:100])
        cir3 = torch.from_numpy(cir3).type(torch.FloatTensor)

        cir4 = np.loadtxt(os.path.join(self.cir4, self.cir4_set[int(self.idxlist[idx]-1)]),delimiter=',',dtype=str)
        cir4 = np.vectorize(lambda x: complex(x.replace('i', 'j')))(cir4)
        cir4 = mmx(np.abs(cir4)[:,0:100])
        cir4 = torch.from_numpy(cir4).type(torch.FloatTensor)

        cir5 = np.loadtxt(os.path.join(self.cir5, self.cir5_set[int(self.idxlist[idx]-1)]),delimiter=',',dtype=str)
        cir5 = np.vectorize(lambda x: complex(x.replace('i', 'j')))(cir5)
        cir5 = mmx(np.abs(cir5)[:,0:100])
        cir5 = torch.from_numpy(cir5).type(torch.FloatTensor)

        cir6 = np.loadtxt(os.path.join(self.cir6, self.cir6_set[int(self.idxlist[idx]-1)]),delimiter=',',dtype=str)
        cir6 = np.vectorize(lambda x: complex(x.replace('i', 'j')))(cir6)
        cir6 = mmx(np.abs(cir6)[:,0:100])
        cir6 = torch.from_numpy(cir6).type(torch.FloatTensor)

        cir7 = np.loadtxt(os.path.join(self.cir7, self.cir7_set[int(self.idxlist[idx]-1)]),delimiter=',',dtype=str)
        cir7 = np.vectorize(lambda x: complex(x.replace('i', 'j')))(cir7)
        cir7 = mmx(np.abs(cir7)[:,0:100])
        cir7 = torch.from_numpy(cir7).type(torch.FloatTensor)

        cir8 = np.loadtxt(os.path.join(self.cir8, self.cir8_set[int(self.idxlist[idx]-1)]),delimiter=',',dtype=str)
        cir8 = np.vectorize(lambda x: complex(x.replace('i', 'j')))(cir8)
        cir8 = mmx(np.abs(cir8)[:,0:100])
        cir8 = torch.from_numpy(cir8).type(torch.FloatTensor)


#### PDP ####

        pdp3 = torch.abs(cir3)**2
        pdp4 = torch.abs(cir4)**2
        pdp5 = torch.abs(cir5)**2
        pdp6 = torch.abs(cir6)**2
        pdp7 = torch.abs(cir7)**2
        pdp8 = torch.abs(cir8)**2


        pdp = torch.cat([pdp3,pdp4],dim=1)#,pdp6,pdp8
        traj = torch.from_numpy(np.loadtxt(os.path.join(self.Traj, self.Traj_set[int(self.idxlist[idx]-1)]),delimiter=',')).type(torch.FloatTensor)
        # traj = torch.transpose(traj,1,0)


        return pdp, traj

