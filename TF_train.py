import tqdm
from torch.utils.data import DataLoader
import DFModel
from misc import *
from Data_utils import *
import torch.optim as optim

# 这是度量学习的一个第三方库
from pytorch_metric_learning import miners, losses, samplers, reducers





if __name__ == '__main__':
    model = DFModel.DFModel(64)
    # mode有两种origin和new，origin就是TF原本的实现，我按照他的代码写的损失函数以及选三元组的方法
    # new是使用pytorch_metric_learning库里的损失函数（circle loss）以及他的选取三元组的方法
    # 两者效果差不多，后者收敛快一点
    mode = 'origin'
    isTrain = True
    num_shot = 5
    num_test = 70
    train_path = r'/home/zhaxuyang/TF/memory_tf/dataset/extracted_AWF775'
    test_path = r'/home/zhaxuyang/TF/memory_tf/dataset/extracted_AWF100'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if mode == 'origin' and isTrain:
        # 训练阶段
        model.to(device)
        EM = MemoryBank(775, 64, device)
        Xa_train, Xp_train, all_traces_train_idx, id_to_classid, data, label, test_data, test_label = loadData(train_path,
                                                                                                               isTrain)
        opt = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6, nesterov=True)
        train_loader = DataLoader(OriginDataset(Xa_train, Xp_train), batch_size=128, shuffle=True)
        loss = OriginLossfuc()
        for epoch in range(30):
            train_loss = 0
            b_id = 0
            if epoch == 0:
                sims = None
            else:
                sims = build_similarities(model, data, label)
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
            model.train()
            for i, (a, p) in loop:
                opt.zero_grad()
                n = build_negatives(a, p, sims, all_traces_train_idx, id_to_classid)
                x_a, x_p, x_n = data[torch.tensor(a).flatten()].to(device), data[torch.tensor(p).flatten()].to(device), \
                    data[torch.tensor(n).flatten()].to(device)
                x = torch.cat((x_a, x_p, x_n), dim=0)
                em = model(x)
                em = torch.chunk(em, 3, dim=0)
                x_a_em = em[0]
                x_p_em = em[1]
                x_n_em = em[2]
                l = loss(x_a_em, x_p_em, x_n_em)
                l.backward()
                opt.step()
                train_loss += l.item()
                b_id += 1
                loop.set_description(f'Epoch [{epoch}/30]')
                loop.set_postfix(loss_now=l.item(), loss=train_loss / b_id)
            torch.save(model.state_dict(), 'train_ori_tri_tf.pth')
    elif mode == 'origin' and not isTrain:
        state_dict = torch.load('train_ori_tri_tf.pth')
        model.load_state_dict(state_dict)
        Xa_train, Xp_train, all_traces_train_idx, id_to_classid, data, label, test_data, test_label = loadData(
            test_path, isTrain, num_shot, num_test)
        train_loader = DataLoader(Dataset(data, label), batch_size=100, shuffle=False)
        test_loader = DataLoader(Dataset(test_data, test_label), batch_size=100, shuffle=False)
        EM = MemoryBank(100, 64, device)
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                EM.update(outputs, targets)
        prototype = EM.getEM()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # nn
                sims = cal_cosine_similarity(outputs, prototype)
                result = torch.argmax(sims, dim=1)
                result = result.eq(targets).sum()
                acc_meter.update(result, targets.size(0))
        print('train_acc: {:.4f}'.format(float(acc_meter.avg * 100)))

        acc_meter.reset()
        with torch.no_grad():
            b_id = 0
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                sims = cal_cosine_similarity(outputs, prototype)
                res1 = torch.argmax(sims, dim=1)
                result = res1.eq(targets).sum()
                acc_meter.update(result, targets.size(0))
                b_id = batch_idx
        print('Test set: {}, test_acc: {:.4f}'.format(len(test_loader), float(acc_meter.avg * 100)))
    elif mode == 'new' and isTrain:
        # loss_func = losses.TripletMarginLoss(0.1)
        loss_func = losses.CircleLoss(m=0.1, gamma=100)
        # miner_func = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.HARD,neg_strategy=miners.BatchEasyHardMiner.HARD)
        # miners.TripletMarginMiner()
        Xa_train, Xp_train, all_traces_train_idx, id_to_classid, data, label, test_data, test_label = loadData(
            train_path, isTrain)
        miner_func = miners.MultiSimilarityMiner(0.1)
        samp = samplers.MPerClassSampler(label, m=10)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        EM = MemoryBank(775, 64, device)
        opt = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6, nesterov=True)
        train_loader = DataLoader(Dataset(data, label), sampler=samp, batch_size=200)
        for epoch in range(30):
            model.train()
            train_loss = 0
            b_id = 0
            for i, (x, y) in enumerate(train_loader):
                opt.zero_grad()
                x, y = x.to(device), y.to(device)
                out = model(x)
                l = loss_func(out, y, miner_func(out, y))
                l.backward()
                opt.step()
                train_loss += l.item()
                b_id = i
            print('Epoch:{}, Loss:{:.4f}'.format(epoch, train_loss / (b_id + 1)))
            torch.save(model.state_dict(), 'test_2.pth')
    elif mode == 'new' and not isTrain:
        loss_func = losses.CircleLoss(m=0.1, gamma=100)
        miner_func = miners.MultiSimilarityMiner(0.1)
        EM = MemoryBank(100, 64, device)
        state_dict = torch.load('test_2.pth')
        model.load_state_dict(state_dict)
        Xa_train, Xp_train, all_traces_train_idx, id_to_classid, data, label, test_data, test_label = loadData(
            test_path, isTrain, num_shot, num_test)
        train_loader = DataLoader(Dataset(data, label), batch_size=100, shuffle=False)
        test_loader = DataLoader(Dataset(test_data, test_label), batch_size=100, shuffle=False)
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                EM.update(outputs, targets)
        prototype = EM.getEM()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # nn
                sims = cal_cosine_similarity(outputs, prototype)
                result = torch.argmax(sims, dim=1)
                result = result.eq(targets).sum()
                acc_meter.update(result, targets.size(0))
        print('train_acc: {:.4f}'.format(float(acc_meter.avg * 100)))

        test_loss = 0
        acc_meter.reset()
        with torch.no_grad():
            b_id = 0
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, targets, miner_func(outputs, inputs))
                test_loss += loss.item()
                # nn
                sims = cal_cosine_similarity(outputs, prototype)
                res1 = torch.argmax(sims, dim=1)
                result = res1.eq(targets).sum()
                acc_meter.update(result, targets.size(0))
        print('Test set: {}, test loss: {:.4f}, test_acc: {:.4f}'.format(len(test_loader), test_loss / (b_id + 1),
                                                                         float(acc_meter.avg * 100)))
