# -*- coding: utf-8
import os

import torch
from torch.utils import data
import cv2
import numpy as np
import json

import random
from PIL import Image
from torchvision import transforms

class BVIDataset(data.Dataset):
    def __init__(self, data_dir, json_path, transform, is_train, is_test=False, read_num=5, start_num=5):
        super(BVIDatasetV3, self).__init__()
        with open(json_path, 'r') as f:
            mos_file_content = json.loads(f.read())
            if is_train:
                self.frame_names_ref = mos_file_content['train']['ref']
                self.frame_names_dis = mos_file_content['train']['dis']
                self.score = mos_file_content['train']['mos']
            elif is_test:
                self.frame_names_ref = mos_file_content['test']['ref']
                self.frame_names_dis = mos_file_content['test']['dis']
                self.score = mos_file_content['test']['mos']
            else:
                self.frame_names_ref = mos_file_content['val']['ref']
                self.frame_names_dis = mos_file_content['val']['dis']
                self.score = mos_file_content['val']['mos']

        self.frames_dir = data_dir
        self.transform = transform
        self.length = len(self.score)
        self.read_num = read_num
        self.start_num = start_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        frame_type = ['ref', 'dis']

        frame_score = 1 - torch.FloatTensor(np.array(float(self.score[idx]))) / 100.0

        frame_dir = os.path.join(self.frames_dir, self.frame_names_dis[idx])
        frame_dir_ref = os.path.join(self.frames_dir, self.frame_names_ref[idx])
        frame_rate = frame_dir.split('/')[-2][:-2]

        # print(filename)
        rn = self.read_num
        sn = self.start_num

        frame_index = [int(sn + frame_num * k / rn) for k in range(rn)]
        Filename = []
        frame = {}
        for i_type in frame_type:
            for i in range(rn):
                index = int(frame_index[i])
                if i_type == 'ref':
                    filename = os.path.join(frame_dir_ref, '{:>05d}.png'.format(index))
                    filename_ref1 = os.path.join(frame_dir_ref, '{:>05d}.png'.format(index - 1))
                    filename_ref2 = os.path.join(frame_dir_ref, '{:>05d}.png'.format(index + 1))
                elif i_type == 'dis':
                    filename = os.path.join(frame_dir, '{:>05d}.png'.format(index))
                    filename_ref1 = os.path.join(frame_dir, '{:>05d}.png'.format(index - 1))
                    filename_ref2 = os.path.join(frame_dir, '{:>05d}.png'.format(index + 1))
                    Filename.append(filename)
                read_frame = self._read_frame(filename)
                read_frame_ref1 = self._read_frame(filename_ref1)
                read_frame_ref2 = self._read_frame(filename_ref2)

                if i == 0:
                    frames = torch.cat((read_frame_ref1, read_frame, read_frame_ref2), 1)
                else:
                    frames = torch.cat((frames, read_frame_ref1, read_frame, read_frame_ref2), 1)
            frame[i_type] = frames
            # frame[i_type] = self.transform(frames)

        return frame['ref'], frame['dis'], frame_score, Filename

    def _read_frame(self, filename):
        cap_frame = cv2.imread(filename)
        cap_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB)
        read_frame = transforms.ToTensor()(cap_frame)
        # read_frame = read_frame.permute(2, 0, 1) 
        read_frame = torch.unsqueeze(read_frame, 0)
        transformed_frame = self.transform(read_frame)
        transformed_frame = transformed_frame.permute(1, 0, 2, 3)
        return transformed_frame
