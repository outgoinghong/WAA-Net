import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
import torch.nn.functional as F
from utility import load_HSI, hyperVca, load_data, reconstruction_SADloss ,Data,My_Loss
from utility import plotAbundancesGT, plotAbundancesSimple, plotEndmembersAndGT, reconstruct
import time
import os
import pandas as pd
from scipy.spatial.distance import cosine
start_time = time.time()
import matplotlib.pyplot as plt
from model import CSSAB
from loss import *
import random
from tqdm import tqdm
def set_seed(seed):
    # 设置 Python 的随机种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 为 CUDA 设置随机种子
        torch.cuda.manual_seed(seed)
        # 为所有 GPU 设置随机种子
        torch.cuda.manual_seed_all(seed)
        # 保证每次卷积的算法一致
        torch.backends.cudnn.deterministic = True
        # 关闭 CuDNN 的自动调优功能
        torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Samson': 'Samson',
                'houston': 'houston',
                'moffett': 'moffett',
                'gai2': 'gai2',
                'Urban': 'Urban',
                }
dataset = "moffett"
hsi = load_HSI("Datasets/" + datasetnames[dataset] + ".mat")
data1 = Data(datasetnames[dataset], device)
data = hsi.array()
P = hsi.gt.shape[0]
col = hsi.cols
line = hsi.rows
levels = [1, 2, 3]
L = data.shape[1]
batch_size = 1
num_runs = 1
EPOCH = 500
if dataset == "Samson":
    seed = 4
    drop = 0.01
    learning_rate = 9e-3
    step_size = 35
    gamma = 0.7
    weight_decay = 3e-3
    beta = 1e-3
    a = 1
    b = 0.003
    c = 0.00
if dataset == "houston":
    seed = 50
    a = 1
    b = 0.022
    c = 0.00
    patch = 5
    dim = 200
    drop_out = 0.1
    learning_rate = 9e-4
    step_size = 40
    gamma = 0.8
    weight_decay =  1e-2
    beta =  8e-4
if dataset == "moffett":
    seed =1
    #seed = 2
    drop_out = 0.1
    learning_rate = 5e-3
    step_size = 35
    gamma = 0.5
    weight_decay =9e-4
    beta = 1e-3
    a = 1
    b = 0.05
    c = 0
    patch = 5
    dim = 200
if dataset == "apex":
    seed = 100
    a = 1
    b = 0.00001
    c = 0.00
    patch = 5
    dim = 200
    drop_out = 0.1
    learning_rate = 9e-4
    step_size = 40
    gamma = 0.8
    weight_decay =  5e-4
    beta =  8e-4
if dataset == "gai2":
    seed = 100
    dim = 200
    EPOCH = 300
    drop_out = 0.1
    learning_rate = 0.0001
    step_size = 45
    gamma = 0.8
    weight_decay = 5e-5
#seed = 42
set_seed(seed)
MSE = torch.nn.MSELoss(reduction='mean')
my_loss = My_Loss()
end = []
abu = []
r = []
output_path = 'Results'
method_name = 'WAANet'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

for run in range(1, num_runs + 1):
#for seed in range(0, 200):  # 遍历0-100的随机种子
    #print(f'Processing seed: {seed}')
    # 重置所有随机状态
    #set_seed(seed)
    #run = seed
    print('Start training!', 'run:', run)
    abundance_GT = torch.from_numpy(hsi.abundance_gt) # (h,w,p)

    abundance_GT = torch.reshape(abundance_GT, (col * line, P)).permute(1, 0)

    abundance_GT = torch.reshape(abundance_GT, (P, col, line))

    #z = slect(data)
    original_HSI = torch.from_numpy(data.reshape(col, line, L)) # (h,w,c)

    original_HSI = original_HSI.permute(2, 0, 1)

    original = torch.from_numpy(data.reshape(col, line, L)) # (h,w,c)
    original = original.permute(2, 0, 1)

    image = np.array(original_HSI)

    xiaobo = image.copy()

    endmembers, _, _ = hyperVca(hsi.array().T, P, datasetnames[dataset])
    VCA_endmember = torch.from_numpy(endmembers)
    GT_endmember = hsi.gt.T
    endmember_init = VCA_endmember.unsqueeze(2).unsqueeze(3).float()

    # load data
    train_dataset = load_data(img=original, transform=transforms.ToTensor())
    # Data loader

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    net = CSSAB(P=P, L=L, size=col ).to(device)
    #net1 = HSA(dim=L)
    #model = UNet(n_channels=156, n_classes=3, bilinear=False, spec_type="all")
    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    endmember_path2 = endmember_folder + '/' + endmember_name + 'vca'
    w = 1
    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name
    y = torch.from_numpy(xiaobo)
    #x = y.permute(2, 0, 1)
    #x = y
    y = y.unsqueeze(0)
    y = y.cuda()

    print(y.shape)
    def weights_init(m):
        # nn.init.kaiming_normal_(net.inc.double_conv[0].weight.data)
        # nn.init.kaiming_normal_(net.inc.double_conv[3].weight.data)
        #nn.init.kaiming_normal_(net.encoder[0].weight.data)
        #nn.init.kaiming_normal_(net.encoder[4].weight.data)
        #nn.init.kaiming_normal_(net.encoder[7].weight.data)
        nn.init.kaiming_normal_(net.encoder3[0].weight.data)
        nn.init.kaiming_normal_(net.encoder3[7].weight.data)
        nn.init.kaiming_normal_(net.encoder3[8].weight.data)



        nn.init.kaiming_normal_(net.smooth[0].weight.data)

    net.apply(net.weights_init)
    model_dict = net.state_dict()
    model_dict["decoder.0.weight"] = endmember_init
    net.load_state_dict(model_dict)
    loss_func = nn.MSELoss(reduction='mean')
    lambda_y2 = 0.04
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    TVLoss = TVLossEndmembers()
    for epoch in range(EPOCH):
        for i, x in enumerate(train_loader):
            x = x.cuda()
            net.train().cuda()
            torch.cuda.empty_cache()
            en_abundance, re = net(x)
            #en_abundance = net(x)
            #pred_linear, pred_abun, pred_endm = net(x)
            #total_loss = my_loss(x, pred_linear, pred_endm, pred_aban=pred_abun)
            abundanceLoss = reconstruction_SADloss(x, re)
            abu_neg_error = torch.mean(torch.relu(-en_abundance))
            abu_sum_error = torch.mean((torch.sum(en_abundance, dim=0) - 1) ** 2)
            abu_loss = abu_neg_error + abu_sum_error
            loss_re = beta * loss_func(re, x)
            #loss_re2 = beta * loss_func(re, output_original)
            #abundanceLoss = reconstruction_SADloss(x, reconstruction_result)
            #total_loss = abundanceLoss
            total_loss = a * abundanceLoss
            #total_loss = abundanceLoss + loss_rec
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                """print(ELoss.cpu().data.numpy())"""
                print("Epoch:", epoch, "| loss: %.4f" % total_loss.cpu().data.numpy())
        scheduler.step()

    #en_abundance, reconstruction_result, decoder_weight = net(x)
    en_abundance, re = net(y)
    en_abundance = torch.squeeze(en_abundance)
    en_abundance = torch.reshape(en_abundance, [P, col * line])
    en_abundance = en_abundance.T
    en_abundance = torch.reshape(en_abundance, [col, line, P])
    abundance_GT = torch.reshape(abundance_GT, [P, col * line])
    abundance_GT = abundance_GT.T
    abundance_GT = torch.reshape(abundance_GT, [line, col, P])
    en_abundance = en_abundance.cpu().detach().numpy()
    abundance_GT = abundance_GT.cpu().detach().numpy()
    endmember_hat = net.state_dict()["decoder.0.weight"].cpu().numpy()
    endmember_hat = np.squeeze(endmember_hat)
    endmember_hat = endmember_hat.T
    GT_endmember = GT_endmember.T
    y_hat = reconstruct(en_abundance, endmember_hat)
    RE = np.sqrt(np.mean(np.mean((y_hat - data) ** 2, axis=1)))
    r.append(RE)
    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'A': en_abundance,
                                                                              'E': endmember_hat,
                                                                              })
    plotAbundancesSimple(en_abundance, abundance_GT, abundance_path, abu)
    plotEndmembersAndGT(endmember_hat, GT_endmember, endmember_path, end)

    torch.cuda.empty_cache()

    print('-' * 70)
    current_rmse = abu[3]  # 假设RE是当前轮的RMSE
    current_sad = end  # 假设end是当前轮的SAD
    # 保存当前结果
    results = []
    current_sad_avg = np.mean(current_sad) if isinstance(current_sad, list) else current_sad
    results.append({
        'seed': seed,
        'rmse': current_rmse,
        'sad': current_sad_avg
    })

end_time = time.time()
end = np.reshape(end, (-1, P + 1))
abu = np.reshape(abu, (-1, P + 1))
dt = pd.DataFrame(end)
dt2 = pd.DataFrame(abu)
dt3 = pd.DataFrame(r)
dt.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各端元SAD及mSAD运行结果.csv')
dt2.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各丰度图RMSE及mRMSE运行结果.csv')
dt3.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '重构误差RE运行结果.csv')
abundanceGT_path = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '参照丰度图'
plotAbundancesGT(hsi.abundance_gt, abundanceGT_path)
print('程序运行时间为:', end_time - start_time, 's')
#endmember_hat = endmember_hat.permute(1, 0)
