import torch.nn as nn
import os
import random
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader


def loadData(path, isTrain, num_shot=5, num_test=70):
    """
    这个方法用于读取数据，很多操作是TF用于选三元组的方法，可以仔细看看；如果使用new模式就不需要手动选三元组
    """
    label2id = {}
    data = []
    classid2id = {}
    label = []
    test_data = []
    test_label = []
    i = 0
    for dir in os.listdir(path):
        label2id[dir] = i
        classid2id[i] = []
        j = 0
        items = []
        for item in os.listdir(path + '/' + dir):
            items.append(item)
        random.shuffle(items)
        for item in items:
            with open(path + '/' + dir + '/' + item, 'rb') as f:
                p = pickle.load(f, encoding='iso-8859-1')
                if isTrain:
                    if j < 25:
                        data.append(p)
                        label.append(i)
                        classid2id[i].append(label.__len__() - 1)
                    elif j < 75:
                        test_data.append(p)
                        test_label.append(i)
                    j += 1
                else:
                    if j < num_shot:
                        data.append(p)
                        label.append(i)
                        classid2id[i].append(label.__len__() - 1)
                    elif j < num_shot + num_test:
                        test_data.append(p)
                        test_label.append(i)
                    j += 1
        i += 1
    id_to_classid = {v: c for c, traces in classid2id.items() for v in traces}
    data = torch.from_numpy(np.array(data)).unsqueeze(1).float()
    label = torch.from_numpy(np.array(label))

    test_data = torch.from_numpy(np.array(test_data)).unsqueeze(1).float()
    test_label = torch.from_numpy(np.array(test_label))

    Xa_train, Xp_train = build_positive_pairs(range(0, 775), classid2id)

    # Gather the ids of all network traces that are used for training
    # This just union of two sets set(A) | set(B)
    all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
    return Xa_train, Xp_train, all_traces_train_idx, id_to_classid, data, label, test_data, test_label


def build_pos_pairs_for_id(classid, classid2id):  # classid --> e.g. 0
    traces = classid2id[classid]
    # pos_pairs is actually the combination C(10,2)
    # e.g. if we have 10 example [0,1,2,...,9]
    # and want to create a pair [a, b], where (a, b) are different and order does not matter
    # e.g. [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    # (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)...]
    # C(10, 2) = 45
    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i + 1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs


def build_positive_pairs(class_id_range, classid2id):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id, classid2id)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]]  # identity
            listX2 += [pair[1]]  # positive example
    perm = np.random.permutation(len(listX1))
    # random.permutation([1,2,3]) --> [2,1,3] just random
    # random.permutation(5) --> [1,0,4,3,2]
    # In this case, we just create the random index
    # Then return pairs of (identity, positive example)
    # that each element in pairs in term of its index is randomly ordered.
    return np.array(listX1)[perm], np.array(listX2)[perm]


def intersect(a, b):
    return list(set(a) & set(b))


def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, id_to_classid, num_retries=50):
    # If no similarities were computed, return a random negative
    if similarities is None:
        return np.random.choice(neg_imgs_idx, len(anc_idxs))
    final_neg = []
    # for each positive pair
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anc_idx, pos_idx = anc_idx.item(), pos_idx.item()
        anchor_class = id_to_classid[anc_idx]
        # positive similarity
        sim = similarities[anc_idx, pos_idx]
        # find all negatives which are semi(hard)
        possible_ids = np.where((similarities[anc_idx] + 0.1) > sim)[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg


def build_similarities(conv, all_imgs, label):
    sim_loader = DataLoader(Dataset(all_imgs, label), batch_size=100, shuffle=False)
    conv.eval()
    emb = torch.tensor([]).cuda()
    with torch.no_grad():
        for i, (x, y) in enumerate(sim_loader):
            x = x.cuda()
            embs = conv(x)
            emb = torch.cat((emb, embs), dim=0)

    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    all_sims = torch.matmul(emb, emb.t()).to('cpu')
    return all_sims


class Dataset():
    def __init__(self, data, label):
        self.X = data
        self.Y = label

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class OriginDataset(torch.utils.data.Dataset):
    def __init__(self, anc, pos):
        self.anc = anc
        self.pos = pos

    def __getitem__(self, index):
        return self.anc[index], self.pos[index]

    def __len__(self):
        return len(self.anc)
