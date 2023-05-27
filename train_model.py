import time

from peak_tract_dataloader import DataLoaderTraining
import numpy as np
import torch
import argparse
from unet_pytorch_deepsup import UNet_Pytorch_DeepSup,UNet_Pytorch_DeepSup_IFT

from torch.optim import Adamax
import torch.nn as nn
import os


def val_f1_score_macro(y_true, y_pred):
    """
    Macro f1. Same results as sklearn f1 macro.

    Args:
        y_true: (n_samples, n_classes)
        y_pred: (n_samples, n_classes)

    Returns:

    """
    f1s = []
    if len(y_true.shape) == 4:
        for i in range(y_true.shape[-1]):
            intersect = np.sum(y_true[:, :, :, i] * y_pred[:, :, :, i])  # works because all multiplied by 0 gets 0
            denominator = np.sum(y_true[:, :, :, i]) + np.sum(
                y_pred[:, :, :, i])  # works because all multiplied by 0 gets 0
            f1 = (2 * intersect) / (denominator + 1e-6)
            f1s.append(f1)
    elif len(y_true.shape) == 3:
        for i in range(y_true.shape[-1]):
            intersect = np.sum(y_true[:, :, i] * y_pred[:, :, i])  # works because all multiplied by 0 gets 0
            denominator = np.sum(y_true[:, :, i]) + np.sum(y_pred[:, :, i])  # works because all multiplied by 0 gets 0
            f1 = (2 * intersect) / (denominator + 1e-6)
            f1s.append(f1)
    return np.array(f1s)


def model_train(args, data_loader, optimizer, train_subjects, start_epoch=0):

    batch_size = args.batch_size
    epoch = args.epoch
    batch_gen_train = data_loader.get_batch_generator(train_subjects, type="train")
    iteration = 0
    nr_batches = int(144 / batch_size)

    for epoch_nr in range(start_epoch+1, epoch + 1):
        weight_factor = float(args.LOSS_WEIGHT)-(args.LOSS_WEIGHT-1)*(epoch_nr/float(args.LOSS_WEIGHT_LEN))
        print("weight_factor =",weight_factor)
        # train
        start_time = time.time()
        local_time=time.asctime(time.localtime(start_time))
        print(local_time)
        for batch_index in range(len(train_subjects)):
            print("-" * 20)
            # for type in model_type:

            for i in range(nr_batches):
                    batch = next(batch_gen_train)
                    x = batch["data"]  # (bs, nr_of_channels, x, y)
                    y = batch["seg"]  # (bs, nr_of_classes, x, y)
                    direction= batch["slice_dir"]
                    subject_index = batch["subject_index"]
                    x = x.to(device)
                    y = y.to(device)

                    outputs = model(x)
                    bce_weight = torch.ones((y.shape[0],y.shape[1],y.shape[2],y.shape[3],)).cuda()
                    bundle_mask = y>0
                    bce_weight[bundle_mask.data] *=weight_factor
                    bce_loss = nn.BCEWithLogitsLoss(weight=bce_weight)(outputs, y)

                    loss = bce_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    outputs_sigmoid = torch.sigmoid(outputs.detach())
                    predict = torch.where(outputs_sigmoid.detach() > 0.5, torch.ones_like(outputs_sigmoid), torch.zeros_like(outputs_sigmoid))

                    iteration += 1
                    print("Train: epoch {} batch {} tract_name {} iteration {} direction {} loss = {:.4f} ".format(epoch_nr,batch_index+1, train_subjects[subject_index], iteration, direction,float(loss)))

                    f1_macro=val_f1_score_macro(predict.cpu().numpy().transpose(0,2,3,1), y.detach().cpu().numpy().transpose(0,2,3,1))

                    print("f1_macro = {:.4f}".format(np.nanmean(f1_macro)))
                    print("f1_per_class =")

                    for f1_index in range(len(f1_macro)):
                      if (f1_index+1)%12==0 :
                        print("{:.4f}".format(f1_macro[f1_index]),end="\n")
                      else:
                        print("{:.4f}".format(f1_macro[f1_index]),end=" ")



        # save model
        if epoch_nr % 5 == 0:
              torch.save(model, args.ckpt_dir +  "/{}_{}_epoch_{}.pth".format(args.action[6:],args.ratio,epoch_nr))
              print("save model " + args.ckpt_dir +"/{}_{}_epoch_{}.pth".format(args.action[6:],args.ratio,epoch_nr))
              os.system("nvidia-smi")
        end_time=time.time()
        run_time=end_time-start_time
        hour = int(run_time/3600)
        minute = int((run_time-hour*3600)/60)
        print("epoch {} running time {}h {}min".format(epoch_nr,hour,minute))

def model_train_novel(args,novel_model,base_model, data_loader, optimizer, train_subjects, start_epoch=0):

    batch_size = args.batch_size
    epoch = args.epoch
    base_model.eval()
    batch_gen_train = data_loader.get_batch_generator(train_subjects, type="train")
    iteration = 0
    nr_batches = int(144 / batch_size)
    if args.ratio==1:
        num_base=36
    elif args.ratio==2:
        num_base=48
    elif args.ratio==5:
        num_base=60


    for epoch_nr in range(start_epoch+1, epoch + 1):
        weight_factor = float(args.LOSS_WEIGHT)-(args.LOSS_WEIGHT-1)*(epoch_nr/float(args.LOSS_WEIGHT_LEN))
        print("weight_factor =",weight_factor)
        # train
        start_time = time.time()
        local_time=time.asctime(time.localtime(start_time))
        print(local_time)
        for batch_index in range(len(train_subjects)):
            print("-" * 20)
            # for type in model_type:

            for i in range(nr_batches):
                    batch = next(batch_gen_train)
                    x = batch["data"]  # (bs, nr_of_channels, x, y)
                    y = batch["seg"]  # (bs, nr_of_classes, x, y)
                    direction= batch["slice_dir"]
                    subject_index = batch["subject_index"]

                    x = x.cuda()
                    y = y.cuda()

                    base_out = base_model(x)
                    novel_out = novel_model(x)

                    # consistency loss
                    con_loss = nn.BCELoss()(torch.sigmoid(novel_out[:, :num_base, ...]), torch.sigmoid(base_out))

                    # seg loss
                    bce_weight = torch.ones((y.shape[0],y.shape[1],y.shape[2],y.shape[3],)).cuda()
                    bundle_mask = y>0
                    bce_weight[bundle_mask.data] *=weight_factor
                    seg_loss = nn.BCEWithLogitsLoss(weight=bce_weight)(novel_out[:,num_base:,...], y)

                    loss = con_loss + seg_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    outputs_sigmoid = torch.sigmoid(novel_out[:,num_base:,...].detach())
                    predict = torch.where(outputs_sigmoid.detach() > 0.5, torch.ones_like(outputs_sigmoid), torch.zeros_like(outputs_sigmoid))

                    iteration += 1

                    # print
                    print("Train: epoch {} batch {} tract_name {} iteration {} direction {} loss = {:.4f} ".format(epoch_nr,batch_index+1, train_subjects[subject_index], iteration, direction,float(loss.detach()) ))
                    f1_macro=val_f1_score_macro(predict.cpu().numpy().transpose(0,2,3,1), y.detach().cpu().numpy().transpose(0,2,3,1))
                    print("f1_macro = {:.4f}".format(np.nanmean(f1_macro)))
                    print("f1_per_class =")
                    for f1_index in range(len(f1_macro)):
                      if (f1_index+1)%12==0 :
                        print("{:.4f}".format(f1_macro[f1_index]),end="\n")
                      else:
                        print("{:.4f}".format(f1_macro[f1_index]),end=" ")

        # save model
        if epoch_nr % 5 == 0:
              torch.save(model, args.ckpt_dir + '/' +  "{}_{}_epoch_{}.pth".format(args.action[6:],args.ratio,epoch_nr))
              print("save model " + args.ckpt_dir + '/' +"{}_{}_epoch_{}.pth".format(args.action[6:],args.ratio,epoch_nr))
              os.system("nvidia-smi")

        end_time=time.time()
        run_time=end_time-start_time
        hour = int(run_time/3600)
        minute = int((run_time-hour*3600)/60)
        print("epoch {} running time {}h {}min".format(epoch_nr,hour,minute))

def model_finetune(args,model, data_loader, optimizer, train_subjects, start_epoch=0):

    batch_size = args.batch_size
    epoch = args.epoch

    batch_gen_train = data_loader.get_batch_generator(train_subjects, type="train")

    iteration = 0
    nr_batches = int(144 / batch_size)

    for epoch_nr in range(start_epoch+1, epoch + 1):
        weight_factor = float(args.LOSS_WEIGHT)-(args.LOSS_WEIGHT-1)*(epoch_nr/float(args.LOSS_WEIGHT_LEN))
        print("weight_factor =",weight_factor)

        # train
        start_time = time.time()
        local_time=time.asctime(time.localtime(start_time))
        print(local_time)
        for batch_index in range(len(train_subjects)):
            print("-" * 20)

            for i in range(nr_batches):
                batch = next(batch_gen_train)
                x = batch["data"]  # (bs, nr_of_channels, x, y)
                y = batch["seg"]  # (bs, nr_of_classes, x, y)

                direction= batch["slice_dir"]
                subject_index = batch["subject_index"]

                x = x.to(device)
                y = y.to(device)

                outputs = model(x)

                bce_weight = torch.ones((y.shape[0],y.shape[1],y.shape[2],y.shape[3],)).cuda()
                bundle_mask = y>0
                bce_weight[bundle_mask.data] *=weight_factor
                bce_loss = nn.BCEWithLogitsLoss(weight=bce_weight)(outputs, y)

                loss = bce_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                outputs_sigmoid = torch.sigmoid(outputs.detach())
                predict = torch.where(outputs_sigmoid.detach() > 0.5, torch.ones_like(outputs_sigmoid), torch.zeros_like(outputs_sigmoid))
                iteration += 1

                #print
                print("Train: epoch {} batch {} tract_name {} iteration {} direction {} loss = {:.4f} ".format(epoch_nr,batch_index+1, train_subjects[subject_index], iteration, direction,float(loss.detach()) ))
                f1_macro=val_f1_score_macro(predict.cpu().numpy().transpose(0,2,3,1), y.detach().cpu().numpy().transpose(0,2,3,1))
                print("f1_macro = {:.4f}".format(np.nanmean(f1_macro)))
                print("f1_per_class =")
                for f1_index in range(len(f1_macro)):
                  if (f1_index+1)%12==0 :
                    print("{:.4f}".format(f1_macro[f1_index]),end="\n")
                  else:
                    print("{:.4f}".format(f1_macro[f1_index]),end=" ")

        # save model
        if epoch_nr % 5 == 0:
              torch.save(model,  +  args.ckpt_dir+"/"+"{}_{}_epoch_{}.pth".format(args.action[6:],args.ratio,epoch_nr))
              print("save model " + args.ckpt_dir+"/"+"{}_{}_epoch_{}.pth".format(args.action[6:],args.ratio,epoch_nr))
              os.system("nvidia-smi")

        end_time=time.time()
        run_time=end_time-start_time
        hour = int(run_time/3600)
        minute = int((run_time-hour*3600)/60)
        print("epoch {} running time {}h {}min".format(epoch_nr,hour,minute))

def model_test(args,model, data_loader,  test_subjects):
    print("*" * 40)
    print("Test")
    # subject_idx = random.randint(0,len(val_subjects)-1)
    subjects_dice = []
    subjects_dice_x = []
    subjects_dice_y = []
    subjects_dice_z = []
    nr_batches = int(144 / args.batch_size)
    dice_tract=[]
    for subject_idx in range(len(test_subjects)):

        batch_gen_val = data_loader.get_batch_generator(test_subjects, subject_idx, type="val")

        global_seg_x = torch.tensor([]).to(device)
        global_seg_y = torch.tensor([]).to(device)
        global_seg_z = torch.tensor([]).to(device)

        global_predict_x = torch.tensor([]).to(device)
        global_predict_y = torch.tensor([]).to(device)
        global_predict_z = torch.tensor([]).to(device)

        seg = torch.tensor([]).to(device)

        for i in range(nr_batches):
            batch = next(batch_gen_val)

            # "data_x", "seg_x", "data_y", "seg_y", "data_z", "seg_z",
            data_x = batch["data_x"]
            seg_x = batch["seg_x"]
            data_y = batch["data_y"]
            seg_y = batch["seg_y"]
            data_z = batch["data_z"]
            seg_z = batch["seg_z"]

            if i == 0:
                seg = seg_x.to(device)
            else:
                seg = torch.cat((seg, seg_x.to(device)), dim=0)

            with torch.no_grad():
                data_x = data_x.to(device)
                seg_x = seg_x.to(device)
                data_y = data_y.to(device)
                seg_y = seg_y.to(device)
                data_z = data_z.to(device)
                seg_z = seg_z.to(device)

                outputs_x = model(data_x)
                outputs_x = torch.sigmoid(outputs_x)

                outputs_y = model(data_y)
                outputs_y = torch.sigmoid(outputs_y)

                outputs_z = model(data_z)
                outputs_z = torch.sigmoid(outputs_z)

                if i == 0:
                    global_predict_x = outputs_x
                    global_predict_y = outputs_y
                    global_predict_z = outputs_z

                    global_seg_x = seg_x
                    global_seg_y = seg_y
                    global_seg_z = seg_z

                else:
                    global_predict_x = torch.cat((global_predict_x, outputs_x), dim=0)
                    global_predict_y = torch.cat((global_predict_y, outputs_y), dim=0)
                    global_predict_z = torch.cat((global_predict_z, outputs_z), dim=0)
                    global_seg_x = torch.cat((global_seg_x, seg_x),dim=0)
                    global_seg_y = torch.cat((global_seg_y, seg_y), dim=0)
                    global_seg_z = torch.cat((global_seg_z, seg_z), dim=0)

        seg = seg.permute(0, 2, 3, 1)

        # (bs,channel,y,z)->(bs,y,z,channel)
        global_predict_x = global_predict_x.permute(0, 2, 3, 1)
        # (bs,channel,x,z)->(x,bs,z,channel)
        global_predict_y = global_predict_y.permute(2, 0, 3, 1)
        # (bs,channel,x,y)->(x,y,bs,channel)
        global_predict_z = global_predict_z.permute(2, 3, 0, 1)

        predict_three_batch = torch.add(torch.add(global_predict_x, global_predict_y), global_predict_z) / 3.0

        predict_three_batch = torch.where(predict_three_batch > 0.5, torch.ones_like(predict_three_batch),
                                          torch.zeros_like(predict_three_batch))
        f1 = val_f1_score_macro(predict_three_batch.cpu().numpy(), seg.cpu().numpy())
        dice_tract.append(f1)
        epoch_dice = np.nanmean(f1)
        print("Num.", subject_idx, "subject =", test_subjects[subject_idx], "dice =", epoch_dice)

        global_predict_x_where = torch.where(global_predict_x > 0.5, torch.ones_like(global_predict_x),
                                             torch.zeros_like(global_predict_x))
        global_predict_y_where = torch.where(global_predict_y > 0.5, torch.ones_like(global_predict_y),
                                             torch.zeros_like(global_predict_y))
        global_predict_z_where = torch.where(global_predict_z > 0.5, torch.ones_like(global_predict_z),
                                             torch.zeros_like(global_predict_z))
        epoch_dice_x = np.nanmean(val_f1_score_macro(global_predict_x_where.cpu().numpy(), seg.cpu().numpy()))
        epoch_dice_y = np.nanmean(val_f1_score_macro(global_predict_y_where.cpu().numpy(), seg.cpu().numpy()))
        epoch_dice_z = np.nanmean(val_f1_score_macro(global_predict_z_where.cpu().numpy(), seg.cpu().numpy()))

        print("dice_x =", epoch_dice_x)
        print("dice_y =", epoch_dice_y)
        print("dice_z =", epoch_dice_z)
        subjects_dice.append(epoch_dice)
        subjects_dice_x.append(epoch_dice_x)
        subjects_dice_y.append(epoch_dice_y)
        subjects_dice_z.append(epoch_dice_z)
    dice_tract=np.array(dice_tract)
    dice_tract=np.mean(dice_tract,axis=0)
    average_subject_dice = float(sum(subjects_dice) / len(subjects_dice))
    average_subject_dice_x = float(sum(subjects_dice_x) / len(subjects_dice_x))
    average_subject_dice_y = float(sum(subjects_dice_y) / len(subjects_dice_y))
    average_subject_dice_z = float(sum(subjects_dice_z) / len(subjects_dice_z))
    print("mean dice =", average_subject_dice)
    print("mean dice_x =", average_subject_dice_x)
    print("mean dice_y =", average_subject_dice_y)
    print("mean dice_z =", average_subject_dice_z)
    print(dice_tract)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    tract_name = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                  'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
                  'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
                  'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
                  'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
                  'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
                  'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
                  'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
                  'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
                  'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

    # base:novel = 60:12 = 5:1
    base_tract_name_60 = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5',
                        'CC_6','CC_7', 'CG_left', 'CG_right', 'MLF_left', 'MLF_right','FX_left', 'FX_right', 'ICP_left',
                        'ICP_right', 'IFO_left', 'IFO_right','MCP', 'SCP_left', 'SCP_right','SLF_I_left', 'SLF_I_right',
                        'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right','STR_left', 'STR_right', 'CC',
                        'T_PREF_left', 'T_PREF_right', 'T_PREM_left','T_PREM_right', 'T_PREC_left', 'T_PREC_right',
                        'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left','T_PAR_right', 'T_OCC_left', 'T_OCC_right',
                        'ST_FO_left', 'ST_FO_right', 'ST_PREF_left','ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right',
                        'ST_PREC_left', 'ST_PREC_right','ST_POSTC_left','ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right',
                        'ST_OCC_left', 'ST_OCC_right']
    novel_tract_name_12 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right', 'OR_left',
                        'OR_right', 'POPT_left', 'POPT_right', 'UF_left', 'UF_right', ]

    # base:novel = 48:24 = 2:1
    base_tract_name_48 = ['CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'FX_left', 'FX_right',
                          'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'T_PREC_left', 'T_PREC_right',
                          'T_POSTC_left', 'T_POSTC_right','T_PAR_left', 'T_PAR_right', 'T_OCC_left', 'T_OCC_right',
                          'ST_FO_left', 'ST_FO_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right',
                          'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PAR_left',
                          'ST_PAR_right', 'ST_OCC_left','ST_OCC_right', 'AF_left', 'AF_right', 'ATR_left', 'ATR_right',
                          'CG_left', 'CG_right', 'MLF_left', 'MLF_right','IFO_left', 'IFO_right', 'STR_left', 'STR_right']
    novel_tract_name_24 = ['ICP_left', 'ICP_right', 'MCP', 'SCP_left', 'SCP_right', 'SLF_I_left',
                           'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right', 'CC',
                           'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right',
                           'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'UF_left', 'UF_right', ]

    # base:novel = 36:36 = 1:1
    base_tract_name_36 = ['CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'FX_left', 'FX_right',
                          'T_PREF_left', 'T_PREF_right','T_PREM_left', 'T_PREM_right', 'T_PREC_left', 'T_PREC_right',
                          'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left', 'T_PAR_right', 'T_OCC_left', 'T_OCC_right',
                          'ST_FO_left', 'ST_FO_right','ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right',
                          'ST_PREC_left','ST_PREC_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right',
                          'ST_OCC_left','ST_OCC_right']
    novel_tract_name_36 = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CG_left', 'CG_right',
                           'MLF_left', 'MLF_right', 'IFO_left', 'IFO_right', 'STR_left', 'STR_right',
                           'ICP_left', 'ICP_right', 'MCP', 'SCP_left', 'SCP_right', 'SLF_I_left',
                           'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right', 'CC',
                           'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right',
                           'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'UF_left', 'UF_right', ]


    base_subjects = ['992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367',
                      '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447',
                      '910241', '907656', '904044', \
                      '901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579',
                      '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456',
                      '859671', '857263', '856766', \
                      '849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653',
                      '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370',
                      '771354', '770352', '765056']
    novel_subjects = ['761957']
    test_subjects = ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754',
                     '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671',
                     ]

    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=48)
    parse.add_argument("--lr", type=float, default=0.002)
    parse.add_argument("--epoch", type=int, help='num of epoch', default=200)
    parse.add_argument("--data_dir", type=str, default="/home/hao/data/MSMT_CSD/")
    parse.add_argument("--label_dir", type=str, default="/home/hao/data/HCP105_Zenodo_NewTrkFormat/")
    parse.add_argument("--action", type=str,  default="train_base_tract")
    parse.add_argument("--ratio", type=int, help = "ratio of base to novel", default=1)
    parse.add_argument("--LOSS_WEIGHT", type=int, default=10)
    parse.add_argument("--LOSS_WEIGHT_LEN", type=int, default=200)
    parse.add_argument("--ckpt_dir", type=str)

    args = parse.parse_args()
    device = torch.device("cuda")



    if args.action == "train_base_tract":

        if args.ratio == 1:
            model = UNet_Pytorch_DeepSup(9, 36).cuda()
            data_loader = DataLoaderTraining(args, base_tract_name_36)
        elif args.ratio == 2:
            model = UNet_Pytorch_DeepSup(9, 48).cuda()
            data_loader = DataLoaderTraining(args, base_tract_name_48)
        elif args.ratio == 5:
            model = UNet_Pytorch_DeepSup(9, 60).cuda()
            data_loader = DataLoaderTraining(args, base_tract_name_60)

        optimizer = Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model_train(args, data_loader, optimizer, novel_subjects,  start_epoch=0)

    elif args.action == "train_novel_tract":

        if args.ratio == 1:
            train_tract_name = novel_tract_name_36
        elif args.ratio == 2:
            train_tract_name = novel_tract_name_24
        elif args.ratio == 5:
            train_tract_name = novel_tract_name_12
        data_loader = DataLoaderTraining(args, train_tract_name)

        base_model = torch.load(args.ckpt_dir+"/"+ "base_tract"+"_"+args.ratio+"_"+"epoch_200.pth")
        novel_model = UNet_Pytorch_DeepSup(9, 72)

        base_model_state = base_model.state_dict()
        novel_model_state = novel_model.state_dict()
        output_layer_name = ['output_2.weight', 'output_2.bias', 'output_3.weight', 'output_3.bias', 'conv_5.weight',
                             'conv_5.bias']

        # get base and novel model layer name
        base_model_layer_name = []
        for k, v in base_model_state.items():
            base_model_layer_name.append(k)

        novel_model_layer_name = []
        for k, v in novel_model_state.items():
            novel_model_layer_name.append(k)

        # novel tract model initialization
        # directly initialize if not output layer, or initialize corresponding channel
        for k, v in novel_model_state.items():
            if k not in output_layer_name:
                novel_model_state[k] = base_model_state[k]

        if args.ratio == 1:
            for l in range(len(output_layer_name)):
                novel_model_state[output_layer_name[l]][:36] = base_model_state[output_layer_name[l]]
        elif args.ratio == 2:
            for l in range(len(output_layer_name)):
                novel_model_state[output_layer_name[l]][:48] = base_model_state[output_layer_name[l]]
        elif args.ratio == 5:
            for l in range(len(output_layer_name)):
                novel_model_state[output_layer_name[l]][:60] = base_model_state[output_layer_name[l]]

        novel_model.load_state_dict(novel_model_state, strict=False)

        # Frozen base tract model
        for k, v in base_model.named_parameters():
            v.requires_grad = False

        base_model = base_model.cuda()
        novel_model = novel_model.cuda()
        optimizer = Adamax(novel_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model_train_novel(args,novel_model,base_model, data_loader, optimizer, novel_subjects,  start_epoch=0)

    elif args.action == "train_CFT":

        if args.ratio == 1:
            train_tract_name = novel_tract_name_36
            CFT_model = UNet_Pytorch_DeepSup(9, 36)
        elif args.ratio == 2:
            train_tract_name = novel_tract_name_24
            CFT_model = UNet_Pytorch_DeepSup(9, 24)
        elif args.ratio == 5:
            train_tract_name = novel_tract_name_12
            CFT_model = UNet_Pytorch_DeepSup(9, 12)

        base_model = torch.load(args.ckpt_dir + "/" + "base_tract" + "_" + args.ratio + "_" + "epoch_200.pth")


        base_model_state = base_model.state_dict()
        CFT_model_state = CFT_model.state_dict()

        layer_name = ['output_2.weight', 'output_2.bias', 'output_3.weight', 'output_3.bias', 'conv_5.weight',
                      'conv_5.bias']

        for k, v in CFT_model_state.items():
            if k not in layer_name:
                CFT_model_state[k] = base_model_state[k]

        CFT_model.load_state_dict(CFT_model_state, strict=False)

        for k, v in CFT_model.named_parameters():
            if k not in layer_name:
                v.requires_grad = False

        data_loader = DataLoaderTraining(args, train_tract_name)
        optimizer = Adamax(CFT_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        CFT_model = CFT_model.cuda()
        model_finetune(args, CFT_model, data_loader,optimizer,novel_subjects, start_epoch=0)

    elif args.action == "train_IFT":

        if args.ratio == 1:
            train_tract_name = novel_tract_name_36
            IFT_model = UNet_Pytorch_DeepSup_IFT(9, 36, dim=36)
        elif args.ratio == 2:
            train_tract_name = novel_tract_name_24
            IFT_model = UNet_Pytorch_DeepSup_IFT(9, 24, dim=48)
        elif args.ratio == 5:
            train_tract_name = novel_tract_name_12
            IFT_model = UNet_Pytorch_DeepSup_IFT(9, 12, dim=60)
        base_model = torch.load(args.ckpt_dir+"/base_tract_epoch_200.pth")
        base_model_state = base_model.state_dict()
        IFT_model_state = IFT_model.state_dict()

        finetune_layer_name = ['conv_6.weight','conv_6.bias']
        for k, v in IFT_model_state.items():
            if k not in finetune_layer_name:
                IFT_model_state[k] = base_model_state[k]
        IFT_model.load_state_dict(IFT_model_state, strict=False)

        for k, v in IFT_model.named_parameters():
            if k not in finetune_layer_name:
                v.requires_grad = False

        data_loader = DataLoaderTraining(args, train_tract_name)
        optimizer = Adamax(IFT_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        IFT_model = IFT_model.cuda()
        model_finetune(args, IFT_model,  data_loader, optimizer, novel_subjects,start_epoch=0)

    elif args.action == "train_TractSeg":
        if args.ratio == 1:
            model = UNet_Pytorch_DeepSup(9, 36).cuda()
            data_loader = DataLoaderTraining(args, novel_tract_name_36)
        elif args.ratio == 2:
            model = UNet_Pytorch_DeepSup(9, 24).cuda()
            data_loader = DataLoaderTraining(args, novel_tract_name_24)
        elif args.ratio == 5:
            model = UNet_Pytorch_DeepSup(9, 12).cuda()
            data_loader = DataLoaderTraining(args, novel_tract_name_12)

        optimizer = Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model_train(args, data_loader, optimizer, novel_subjects, start_epoch=0)

    elif args.action == "test_novel_tract":

        if args.ratio == 1:
            test_tract_name = novel_tract_name_36
            novel_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))
        elif args.ratio == 2:
            test_tract_name = novel_tract_name_24
            novel_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))
        elif args.ratio == 5:
            test_tract_name = novel_tract_name_12
            novel_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))

        data_loader = DataLoaderTraining(args, test_tract_name)
        model_test(args,novel_model,data_loader,test_subjects)

    elif args.action == "test_CFT":

        if args.ratio == 1:
            test_tract_name = novel_tract_name_36
            CFT_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))
        elif args.ratio == 2:
            test_tract_name = novel_tract_name_24
            CFT_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))
        elif args.ratio == 5:
            test_tract_name = novel_tract_name_12
            CFT_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))

        data_loader = DataLoaderTraining(args, test_tract_name)
        model_test(args,CFT_model,data_loader,test_subjects)

    elif args.action == "test_IFT":

        if args.ratio == 1:
            test_tract_name = novel_tract_name_36
            IFT_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))
        elif args.ratio == 2:
            test_tract_name = novel_tract_name_24
            IFT_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))
        elif args.ratio == 5:
            test_tract_name = novel_tract_name_12
            IFT_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))

        data_loader = DataLoaderTraining(args, test_tract_name)
        model_test(args, IFT_model,data_loader,test_subjects)
    elif args.action == "test_TractSeg":

        if args.ratio == 1:
            test_tract_name = novel_tract_name_36
            TractSeg_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))
        elif args.ratio == 2:
            test_tract_name = novel_tract_name_24
            TractSeg_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))
        elif args.ratio == 5:
            test_tract_name = novel_tract_name_12
            TractSeg_model = torch.load(args.ckpt_dir + "/" + "{}_{}_epoch_200.pth".format(args.action[5:], args.ratio))

        data_loader = DataLoaderTraining(args, test_tract_name)
        model_test(args, TractSeg_model,data_loader,test_subjects)