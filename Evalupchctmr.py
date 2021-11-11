"""Code for training DDFSeg."""
#Peichenhao

from datetime import datetime

import json

import numpy as np

import random

import os

import cv2

import SimpleITK as sitk
import glob
import tensorflow as tf


from skimage import transform
import data_loader, losses
import  model
import nibabel as nib
import cv2

from stats_func import *

def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

checkpoint_path='./output//'
save_DIR='./result'


class PCH:
    """The PCH module."""

    def __init__(self, config):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._source_train_pth = config['source_train_pth']
        self._target_train_pth = config['target_train_pth']
        self._source_val_pth = config['source_val_pth']
        self._target_val_pth = config['target_val_pth']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._pool_size = int(config['pool_size'])
        self._lambda_a = float(config['_LAMBDA_A'])
        self._lambda_b = float(config['_LAMBDA_B'])
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])
        self._base_lr = float(config['base_lr'])
        self._max_step = int(config['max_step'])
        self._keep_rate_value = float(config['keep_rate_value'])
        self._is_training_value = bool(config['is_training_value'])
        self._batch_size = int(config['batch_size'])
        self._lr_gan_decay = bool(config['lr_gan_decay'])
        self._lsgan_loss_p_scheduler = bool(config['lsgan_loss_p_scheduler'])
        self._to_restore = bool(config['to_restore'])
        self._checkpoint_dir = config['checkpoint_dir']

        self.fake_images_A = np.zeros(
            (self._pool_size, self._batch_size, model.IMG_HEIGHT, model.IMG_WIDTH, 1))
        self.fake_images_B = np.zeros(
            (self._pool_size, self._batch_size, model.IMG_HEIGHT, model.IMG_WIDTH, 1))

    def model_setup(self):

        self.input_a = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                3
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                3
            ], name="input_B")
        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_B")
        self.gt_a = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                4
            ], name="gt_A")

        self.keep_rate = tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.global_step = tf.train.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate_gan = tf.placeholder(tf.float32, shape=[], name="lr_gan")
        self.learning_rate_seg = tf.placeholder(tf.float32, shape=[], name="lr_seg")

        self.lr_gan_summ = tf.summary.scalar("lr_gan", self.learning_rate_gan)
        self.lr_seg_summ = tf.summary.scalar("lr_seg", self.learning_rate_seg)

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(inputs, skip=self._skip, is_training=self.is_training, keep_rate=None)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
        self.pred_mask_b = outputs['pred_mask_b']
        self.pred_mask_fake_b = outputs['pred_mask_fake_b']
        self.prob_fea_fake_b_is_real = outputs['prob_fea_fake_b_is_real']
        self.prob_fea_b_is_real = outputs['prob_fea_b_is_real']

        self.prob_real_a_aux = outputs['prob_real_a_aux']
        self.prob_fake_a_aux_is_real = outputs['prob_fake_a_aux_is_real']
        self.prob_fake_pool_a_aux_is_real = outputs['prob_fake_pool_a_aux_is_real']
        self.prob_cycle_a_is_real = outputs['prob_cycle_a_is_real']
        self.prob_cycle_a_aux_is_real = outputs['prob_cycle_a_aux_is_real']

def dice_compute(pred, groundtruth):           #batchsize*channel*W*W
    dice=[]
    for i in range(4):
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]


    return np.array(dice,dtype=np.float32)




def IOU_compute(pred, groundtruth):
    iou=[]
    for i in range(4):
        iou_i = (np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)-np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)
        iou=iou+[iou_i]


    return np.array(iou,dtype=np.float32)



def main(config_filename):
    Lines = ReadTxtName('./ctmrtestct/img/list.txt')
    laLines = ReadTxtName('./ctmrtestct/lab/list.txt')
    rawpath='./ctmrtestct/img/'
    labelpath = './ctmrtestct/lab/'
    with open(config_filename) as config_file:
        config = json.load(config_file)

    pch_model = PCH(config)

    pch_model.model_setup()

    init = (tf.global_variables_initializer(),

            tf.local_variables_initializer())

    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        modellist=['./output/20200705-093821/pch-16799']
        allresult = open('./result/mytest/dice.txt', 'a+')
        for i in modellist:
            arr=[]
            chkpt_fname = tf.train.latest_checkpoint(checkpoint_path)
            print(chkpt_fname)
            modelname = i
            saver.restore(sess, modelname)
            print('finish loading model!')

            total_dice = np.zeros((4,))
            total_Iou = np.zeros((4,))

            sum = 0
            diceall=0
            for i in range(len(Lines)):
                imgpath=rawpath+Lines[i]
                img = nib.load(imgpath)
                img_arr = img.get_data()
                affine = img.affine

                labpath = labelpath+ laLines[i]
                lab = nib.load(labpath)
                lab_arr = lab.get_data()
                affinelab = lab.affine

                img_arr = np.expand_dims(img_arr, -1)
                print(img_arr.shape)
                img_arr = img_arr.astype(np.float32)
                img_arr = np.concatenate((img_arr, img_arr, img_arr), axis=2)
                print(img_arr.shape)
                img_arr = np.expand_dims(img_arr, 0)

                output = np.zeros((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
                output_numpy = sess.run(pch_model.pred_mask_b,
                                        feed_dict={pch_model.input_b: img_arr,
                                                   pch_model.is_training: False})

                print(output_numpy.shape)

                truearg0 = np.argmax(output_numpy, axis=3)
                print(truearg0.shape)
                print(truearg0.shape)
                saveimg = truearg0[0, :, :]
                saveimg = saveimg.astype(np.float32)
                dice = dice_compute(saveimg, lab_arr)
                diceall=diceall+dice
                arr.append(dice)
                savedir=save_DIR+'/mytest/'+ modelname.split('/')[3]
                if not os.path.isdir(savedir):
                    os.makedirs(savedir)
                dicefile=save_DIR+'/mytest/'+ modelname.split('/')[3]+'/result.txt'
                reultev = open(dicefile, 'a+')
                savename = save_DIR+'/mytest/'+ modelname.split('/')[3]+'/'+ laLines[i] + '.nii.gz'
                new = nib.Nifti1Image(saveimg, affine)
                nib.save(new, savename)
                sum = sum + 1
                reultev.write('/Dice/:' + str(dice) + '\n')
            arr = np.delete(arr, 0, axis=1)
            arr_mean = np.mean(arr, axis=0)
            arr_std = np.std(arr, ddof=1, axis=0)
            arr_mean_all = np.mean(arr)
            arr_std_all = np.std(arr, ddof=1)
            print('ALL:')
            allresult.write('/Dice/:'+ modelname + str(diceall / sum) + '\n')
            allresult.write('/Dice/:'+ modelname + str(arr_mean) + '\n')
            allresult.write('/Dice/:'+ modelname + str(arr_std) + '\n')
            allresult.write('/Dice/:'+ modelname + str(arr_mean_all) + '\n')
            allresult.write('/Dice/:' + modelname + str(arr_std_all) + '\n')
            reultev.close()
        allresult.close()

if __name__ == '__main__':
    main(config_filename='./config_param.json')

