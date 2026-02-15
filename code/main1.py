#!/usr/bin/env python
# coding=utf-8
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from PM_Dataset import PM_Dataset
from VTransformer import MultiTaskVT
from Loss import MultiTaskVAELoss

# ----------------------
# 各种超参数
log_file = str(time.time())
device = torch.device("cuda")

root_path = "E:\\PM5G"
E_train = os.path.join(root_path,'Train')
E_test = os.path.join(root_path,'Test')

E_list = os.path.join(root_path,'Test_5_fold')

E_save_log = os.path.join(root_path,'Log')
E_save_model = os.path.join(root_path,'Model')


E_model = os.path.join(E_save_model, 'PM_VT_AP34_260214_1-4.pkl')
E_para = os.path.join(E_train, log_file, '.pth.tar')
epoch_max = 300
test_err_list = [0]*epoch_max
test_loss_list = [0]*epoch_max
test_rmse_list = [0]*epoch_max
test_mae_list = [0]*epoch_max
best_loss = []
best_model = []


def main():
    time_value = 0
    # VT= torch.load(E_model).cuda()
    # VT = MultiTaskVT(hidden_dim=256,nhead=8,num_layers=1,
    #                  global_latent_dim1=16,global_latent_dim2=16,local_latent_dim=16,cir_dim=200).cuda()
    # VT_Loss = MultiTaskVAELoss([0.3,0.3,0.001]).cuda()

    # VT = MultiTaskVT().cuda()
    # VT_Loss = MultiTaskVAELoss([0.3,0.3]).cuda()

    VT = MLP().cuda()
    VT_Loss = nn.MSELoss(reduction='mean').cuda()
    betas = 0.9
    params = 0.999
    lr = 0.0001
    accumulation = 0
    batchsize_train = 32
    batchsize_test = 32
    start_time = time.time()
    best_acc = 1000


    train_dataset = PM_Dataset(E_train,E_list,'Train1-4.txt')
    train_loader = DataLoader(train_dataset, batch_size=batchsize_train, shuffle=True, num_workers=10)


    test_dataset = PM_Dataset(E_train,E_list,'Test1-4.txt')
    test_loader = DataLoader(test_dataset, batch_size=batchsize_test, shuffle=False, num_workers=8)
    log_pointer = open(os.path.join(E_save_log, 'PM_VT_AP34_260214_1-4.txt'), 'w')
    log_pointer.write(log_file + '\n')
    log_pointer.flush()
    train_err = 0
    train_rmse = 0
    train_mae = 0
    for epoch in range(0, epoch_max):
        # ---------------------------------------------------
        # train
        print("*" * 20, 'Train', "*" * 20, epoch, "*" * 10)
        log_pointer.write('********************Train********************' + str(epoch) + '********************'+'\n')


        for iteration, data in enumerate(train_loader, 0):
            VT.train()
            train_x, train_traj = data
            train_x = train_x.type(torch.FloatTensor).cuda()
            train_traj = train_traj.cuda()

## 半监督学习方法的输出格式
            recon_x, p, g1_mu, g1_logvar, g2_mu, g2_logvar, l_mu, l_logvar = VT(train_x)
            loss_train,recon_loss,position_loss,kl_global1,kl_global2,kl_local = VT_Loss(p, train_traj, recon_x, train_x, g1_mu, g1_logvar, g2_mu, g2_logvar, l_mu, l_logvar)


            if accumulation == 0:  # gradient accumulation
                if epoch % 2 == 0:
                    lr = 0.001 * (0.9 ** (epoch/2)) + 0.000005
                optimizer = torch.optim.Adam(VT.parameters(), lr=lr, betas=(betas, params))

                optimizer.zero_grad()  # reset gradient
                loss_train.backward()
                optimizer.step()

            else:
                lr = 0.00005 * math.cos(2 * math.pi * epoch / 20) + 0.00005 + 0.00001
                optimizer = torch.optim.Adam(VT.parameters(), lr=lr, betas=(betas, params))
                if (iteration + 1) % accumulation == 0:
                    optimizer.zero_grad()  # reset gradient
                    optimizer.step()
                    loss_train.backward()
            # ------------------------
            # train accuracy

            # p = train_traj
            current_err = torch.mean(torch.norm(train_traj- p,dim=2))## 平均欧式距离。在x,y的维度。
            train_err = current_err + train_err

            current_rmse = torch.sqrt(torch.mean((train_traj - p) ** 2))
            train_rmse = current_rmse + train_err

            current_mae = torch.mean(torch.abs(train_traj - p))
            train_mae = current_mae + train_mae

            if iteration % 100 == 0:
                log_pointer.write('loss = ' + str(loss_train) + '  epoch = ' + str(epoch) + '  iteration = ' + str(
                    iteration + 1) + '  learn_rate =' + str(lr) + '\n')
                log_pointer.write(
                    '  current train accuracy = ' + str(current_err) + '\n')
                log_pointer.write(
                    '  current train rmse = ' + str(current_rmse) + '\n')
                log_pointer.write(
                    '  current train mae = ' + str(current_mae) + '\n')
                log_pointer.flush()
        train_err = train_err / (iteration+1)
        train_rmse = train_rmse / (iteration+1)
        train_mae = train_mae / (iteration+1)
        print('Epoch %d    train acc = %.03f' % (epoch, train_err))
        print('Epoch %d    train rmse = %.03f' % (epoch, train_rmse))
        print('Epoch %d    train mae = %.03f' % (epoch, train_mae))

        log_pointer.write('Epoch = ' + str(epoch) + '  train acc = ' + str(train_err) + '  train rmse = ' + str(train_rmse) + '  train mae = ' + str(train_mae) + '\n')
        log_pointer.flush()
        torch.cuda.empty_cache()
        # --------
        # test
        print("*" * 20, 'test', "*" * 20)
        log_pointer.write('********************Test********************'+'\n')
        log_pointer.write('********************************************'+'\n')
        log_pointer.flush()

        test_err = 0
        test_rmse = 0
        test_mae = 0
        torch.cuda.empty_cache()
        for iteration, test_data in enumerate(test_loader, 0):
            with torch.no_grad():
                VT.eval()
                test_x, test_traj = test_data
                test_x = test_x.type(torch.FloatTensor).cuda()
                test_traj = test_traj.cuda()

                recon_x, p, g1_mu, g1_logvar, g2_mu, g2_logvar, l_mu, l_logvar = VT(test_x)

            current_err_test = torch.mean(torch.norm(test_traj - p, dim=2))
            test_err = current_err_test + test_err

            current_rmse_test = torch.sqrt(torch.mean((test_traj - p) ** 2))
            test_rmse = current_rmse_test + test_rmse

            current_mae_test = torch.mean(torch.abs(test_traj - p))
            test_mae = current_mae_test + test_mae

            if (iteration + 1) % 1000 == 0:
                print('iteration = ', iteration)
                print('seperate index: ')
                print('positioning accuracy = %0.4f' % current_err_test)
                log_pointer.write('iteration = ' + str(iteration) + '\n')
                log_pointer.write('current test accuracy = ' + str(current_err_test) + '\n')
                log_pointer.flush()
        test_err = test_err/(iteration + 1)
        test_rmse = test_rmse/(iteration + 1)
        test_mae = test_mae/(iteration + 1)

        print('Test accuracy = %0.3f' % test_err)
        print('Test rmse = %0.3f' % test_rmse)
        print('Test mae = %0.3f' % test_mae)

        torch.save(VT,  E_model)
        test_err_list[epoch] = test_err
        test_rmse_list[epoch] = test_rmse
        test_mae_list[epoch] = test_mae

        print('******************************')
        print('**********NEXT EPOCH**********')
        print('******************************')
        log_pointer.writelines(str(test_err)+'\n')
        log_pointer.writelines(str(test_rmse)+'\n')
        log_pointer.writelines(str(test_mae)+'\n')

        log_pointer.write('******************************' + '\n')
        log_pointer.write('**********NEXT EPOCH**********' + '\n')
        log_pointer.write('******************************' + '\n')
        log_pointer.flush()
        # -------
        # end_time = time.clock()
        # print(end_time - start_time)
        end_time = time.time()
        time_value_epoch = (end_time - start_time) / 3600
        print("-" * 80)
        print('Time spend = ', time_value_epoch, 'hour')
        print("-" * 80)
        torch.cuda.empty_cache()
    time_value += time_value_epoch
    log_pointer.close()
    # os.system('shutdown -s -f -t 5')


if __name__ == '__main__':
    main()


