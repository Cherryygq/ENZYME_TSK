'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import csv

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def read_dict(path):
    'Reads Python dictionary stored in a csv file'
    dictionary = {}
    for key, val in csv.reader(open(path)):
        dictionary[key] = val
    return dictionary


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['0', 'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K',
             'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root    ##data_path = 'data/enzyme_data/'
        self.npoints = args.num_point   ##npoint:一张图中点的个数1024
        self.process_data = process_data    ##false
        self.uniform = args.use_uniform_sample  ##false
        self.use_normals = args.use_normals   ##false
        self.num_category = args.num_category   ##6

        # if self.num_category == 10:
        #     self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        # else:
        #     self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.catfile = os.path.join(self.root, 'enzyme_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))##dict(zip(a,b))将两个列表合并成一个字典，a为key,b为value

        # shape_ids = {}
        # if self.num_category == 10:
        #     shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
        #     shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        # else:
        #     shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        #     shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        # shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]##shape_names[0]=airplane
        # self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
        #                  in range(len(shape_ids[split]))]
        self.enzymes = read_dict('./data/newcsv/pre' + split + '_dic.csv')
        # self.enzymes = read_dict('./data/enzyme_data/pre'+split+'_dic.csv')
        self.enzlist = list(self.enzymes)
        print('The size of %s data is %d' % (split, len(self.enzlist)))
        # print('The size of %s data is %d' % (split, len(self.datapath)))

        self.list_of_points = [None] * len(self.enzlist)
        self.list_of_labels = [None] * len(self.enzlist)
        n = 0
        for k in self.enzlist:
            # npzfile = np.load('./data/enzyme_data/npzfile_1024_CA_all/' + str(k) + 'res.npz')
            npzfile = np.load('./data/CA_all_1000/' + str(k) + 'res.npz')
            label = self.enzymes[k]
            label = np.array([label]).astype(np.int32)
            label = label - 1
            a = np.array(pc_normalize(npzfile['res_coords']))
            # a = npzfile['res_coords']
            if(a.shape==0):
                print("shape=0",k)
            b = seq2onehot(npzfile['seqCA'])
            # b = npzfile['res_seq']
            self.list_of_points[n] = np.float32(np.hstack((a, b)))
            self.list_of_labels[n] = label
            n = n + 1



        # if self.uniform:
        #     self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        # else:
        #     self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # if self.process_data:
        #     if not os.path.exists(self.save_path):
        #         print('Processing data %s (only running in the first time)...' % self.save_path)
        #         self.list_of_points = [None] * len(self.datapath)
        #         self.list_of_labels = [None] * len(self.datapath)
        #
        #         for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
        #             fn = self.datapath[index]
        #             cls = self.classes[self.datapath[index][0]]
        #             cls = np.array([cls]).astype(np.int32)
        #             point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        #
        #             if self.uniform:
        #                 point_set = farthest_point_sample(point_set, self.npoints)
        #             else:
        #                 point_set = point_set[0:self.npoints, :]
        #
        #             self.list_of_points[index] = point_set
        #             self.list_of_labels[index] = cls
        #
        #         with open(self.save_path, 'wb') as f:
        #             pickle.dump([self.list_of_points, self.list_of_labels], f)
        #     else:
        #         print('Load processed data from %s...' % self.save_path)
        #         with open(self.save_path, 'rb') as f:
        #             self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        # return len(self.datapath)
        return len(self.enzlist)

    def _get_item(self, index):
        # if self.process_data:
        #     point_set, label = self.list_of_points[index], self.list_of_labels[index]
        # else:
        #     fn = self.datapath[index]##例：datapath[0]=airplane,data\modelnet40_normal_resampled\airplane\airplane_0001.txt
        #     cls = self.classes[self.datapath[index][0]]##self.datapath[0][0]=airplane,cls=classes[airplane]=1
        #     label = np.array([cls]).astype(np.int32)##label=1(int)
        #     point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)##打开TXT文件
        #
        #     if self.uniform:
        #         point_set = farthest_point_sample(point_set, self.npoints)
        #     else:
        #         point_set = point_set[0:self.npoints, :]
        #
        # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        # if not self.use_normals:
        #     point_set = point_set[:, 0:3]
        point_set = self.list_of_points[index]
        # point_set = pc_normalize(self.list_of_points[index])
        label = self.list_of_labels[index]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


# if __name__ == '__main__':
#     import torch
#
#     data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
#     DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
#     for point, label in DataLoader:
#         print(point.shape)
#         print(label.shape)
