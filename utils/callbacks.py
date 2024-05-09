import os

import matplotlib
import torch
import torch.nn.functional as F

import pre_process
import multiprocessing

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU, fast_hist, per_class_iu


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes,name_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.name_classes       =name_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'w') as f:
                f.write(str(0))
                f.write("\n")
            with open(os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"), 'w') as f:
                f.write("epoch\tmiou\t")
                if self.name_classes is not None:
                    for ind_class in range(self.num_classes):
                        f.write(name_classes[ind_class]+"\t")
                f.write("\n")

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")##改
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                # image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")##改
                image_path  = image_id##改
                image       = Image.open(image_path)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_png(image)
                # image.save(os.path.join(pred_dir, image_id + ".png"))##改
                image.save(os.path.join(pred_dir,image_id.rsplit('/',1)[1].replace("jpg","png")))##改
                        
            print("Calculate miou.")
            _, IoUs, PA_Recall, Precision,name_classes = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, self.name_classes)##改  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            
            with open(os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"), 'a') as f:
                f.write(str(epoch)+"\t"+str(round(temp_miou,2))+"\t")
                if name_classes is not None:
                    for ind_class in range(self.num_classes):
                        f.write(str(round(IoUs[ind_class] * 100, 2))+"\t")
                f.write("\n")
            # shutil.copy(os.path.join(self.log_dir, "epoch_miou.txt"),os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"))
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
##batchsize方案
    def get_miou_png_batchsize(self,img_batchsize):
        count=0
        image_res=[]
        if len(img_batchsize)==0:
            return image_res
            
        # batchsize_data=np.empty(shape=(0,self.input_shape[1],self.input_shape[0],3),dtype=np.float32)
        for image in img_batchsize:
            #---------------------------------------------------------#
            #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
            #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
            #---------------------------------------------------------#
            image       = cvtColor(image)
            orininal_h  = np.array(image).shape[0]
            orininal_w  = np.array(image).shape[1]
            #---------------------------------------------------------#
            #   给图像增加灰条，实现不失真的resize
            #   也可以直接resize进行识别
            #---------------------------------------------------------#
            image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
            #---------------------------------------------------------#
            #   添加上batch_size维度
            #---------------------------------------------------------#
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
            if count==0:
                batchsize_data=image_data
            else:
                batchsize_data=np.append(batchsize_data,image_data,axis=0)
            count+=1
        with torch.no_grad():
            images = torch.from_numpy(batchsize_data)
            if self.cuda:
                images = images.cuda()
            res=self.net(images)
            for i in range(len(img_batchsize)):    
                #---------------------------------------------------#
                #   图片传入网络进行预测
                #---------------------------------------------------#
                # pr = self.net(images)[0]
                pr=res[i]
                #---------------------------------------------------#
                #   取出每一个像素点的种类
                #---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
                #--------------------------------------#
                #   将灰条部分截取掉
                #--------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
                #---------------------------------------------------#
                #   进行图片的resize
                #---------------------------------------------------#
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
                #---------------------------------------------------#
                #   取出每一个像素点的种类
                #---------------------------------------------------#
                pr = pr.argmax(axis=-1)
                
                image = Image.fromarray(np.uint8(pr))
                image_res.append(image)
    
        # image = Image.fromarray(np.uint8(pr))
        # return image
        return image_res
    
    def on_epoch_end_batchsize(self, epoch, model_eval,batchsize):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")##改
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            
            interval=len(self.image_ids)//batchsize+1
            id=0
            count=0
            img_batchsizes=[[] for _ in range(interval)]
            for image_id in tqdm(self.image_ids):
                img_batchsizes[id%interval].append(image_id)
                count+=1
                if count>batchsize:
                    count=0
                    id+=1
            hist = np.zeros((self.num_classes, self.num_classes))
            for i in tqdm(range(interval)):#每个batchsize
                #批量读取图片
                img_batchsize=[]
                for image_path in img_batchsizes[i]:
                    img_batchsize.append(Image.open(image_path))
                preds=self.get_miou_png_batchsize(img_batchsize)
                
                #计算miou部分或许可以改为多线程 
                #计算miou
                if len(preds)==0:
                    continue
                for j in range(len(img_batchsizes[i])):
                    png,_=pre_process.generateMask_item(img_batchsizes[i][j],self.num_classes,self.name_classes)        
                    label=np.array(png)#标签
                    pred=np.array(preds[j])
                    # 如果图像分割结果与标签的大小不一样，这张图片就不计算
                    if len(label.flatten()) != len(pred.flatten()):  
                        print(
                            'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}'.format(
                                len(label.flatten()), len(pred.flatten()),img_batchsizes[i][j]))
                        continue
                    hist += fast_hist(label.flatten(), pred.flatten(), self.num_classes) 
            # for image_id in tqdm(self.image_ids):
            #     #-------------------------------#
            #     #   从文件中读取图像
            #     #-------------------------------#
            #     # image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")##改
            #     image_path  = image_id##改
            #     image       = Image.open(image_path)
            #     #------------------------------#
            #     #   获得预测txt
            #     #------------------------------#
            #     image       = self.get_miou_png(image)
            #     # image.save(os.path.join(pred_dir, image_id + ".png"))##改
            #     image.save(os.path.join(pred_dir,image_id.rsplit('/',1)[1].replace("jpg","png")))##改
                        
            # print("Calculate miou.")
            # _, IoUs, PA_Recall, Precision,name_classes = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, self.name_classes)##改  # 执行计算mIoU的函数
            
            IoUs=per_class_iu(hist)
            
            temp_miou = np.nanmean(IoUs) * 100
            self.mious.append(temp_miou)
            self.epoches.append(epoch)
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            
            with open(os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"), 'a') as f:
                f.write(str(epoch)+"\t"+str(round(temp_miou,2))+"\t")
                if self.name_classes is not None:
                    for ind_class in range(self.num_classes):
                        f.write(str(round(IoUs[ind_class] * 100, 2))+"\t")
                f.write("\n")
            # shutil.copy(os.path.join(self.log_dir, "epoch_miou.txt"),os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"))
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
    
    
    def on_epoch_end_batchsize_multithread(self, epoch, model_eval,batchsize,thread_num):
        # if epoch % self.period == 0 and self.eval_flag:
        if self.eval_flag and (epoch==50 or epoch==72 or (epoch>72 and (epoch-72)%self.period==0)):#50轮，72轮，72轮后每5轮评估一次
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")##改
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            
            interval=len(self.image_ids)//batchsize+1
            id=0
            count=0
            img_batchsizes=[[] for _ in range(interval)]
            for image_id in tqdm(self.image_ids):
                img_batchsizes[id%interval].append(image_id)
                count+=1
                if count>batchsize:
                    count=0
                    id+=1
            hist = np.zeros((self.num_classes, self.num_classes))
            for i in tqdm(range(interval)):#每个batchsize
                #批量读取图片
                img_batchsize=[]
                for image_path in img_batchsizes[i]:
                    img_batchsize.append(Image.open(image_path))
                preds=self.get_miou_png_batchsize(img_batchsize)
                
                if len(preds)==0:
                    continue
                #计算miou部分或许可以改为多线程 
                thread_num=min(thread_num,48)#最多开48个线程
                preds_thread=[[] for _ in range(thread_num)]
                img_thread=[[] for _ in range(thread_num)]
                interval_thread=len(preds)/thread_num
                count_thread=0
                id_thread=0
                for k in tqdm(range(len(preds))):
                    preds_thread[id_thread%thread_num].append(preds[k])
                    img_thread[id_thread%thread_num].append(img_batchsizes[i][k])
                    count_thread+=1
                    if count_thread>interval_thread:
                        count_thread=0
                        id_thread+=1
                pool=multiprocessing.Pool(thread_num)
                results=[]
                for t in range(thread_num):
                    results.append(pool.apply_async(func=cal_iou_multithread,args=(preds_thread[t],img_thread[t],self.num_classes,self.name_classes,)))
                pool.close()
                pool.join()
                
                for result in results:
                    hist+=result.get()
                # #计算miou
                # for j in range(len(img_batchsizes[i])):
                #     png,_=pre_process.generateMask_item(img_batchsizes[i][j],self.num_classes,self.name_classes)        
                #     label=np.array(png)#标签
                #     pred=np.array(preds[j])
                #     # 如果图像分割结果与标签的大小不一样，这张图片就不计算
                #     if len(label.flatten()) != len(pred.flatten()):  
                #         print(
                #             'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}'.format(
                #                 len(label.flatten()), len(pred.flatten()),img_batchsizes[i][j]))
                #         continue
                #     hist += fast_hist(label.flatten(), pred.flatten(), self.num_classes) 
            # for image_id in tqdm(self.image_ids):
            #     #-------------------------------#
            #     #   从文件中读取图像
            #     #-------------------------------#
            #     # image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")##改
            #     image_path  = image_id##改
            #     image       = Image.open(image_path)
            #     #------------------------------#
            #     #   获得预测txt
            #     #------------------------------#
            #     image       = self.get_miou_png(image)
            #     # image.save(os.path.join(pred_dir, image_id + ".png"))##改
            #     image.save(os.path.join(pred_dir,image_id.rsplit('/',1)[1].replace("jpg","png")))##改
                        
            # print("Calculate miou.")
            # _, IoUs, PA_Recall, Precision,name_classes = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, self.name_classes)##改  # 执行计算mIoU的函数
            
            IoUs=per_class_iu(hist)
            
            temp_miou = np.nanmean(IoUs) * 100
            self.mious.append(temp_miou)
            self.epoches.append(epoch)
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            
            with open(os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"), 'a') as f:
                f.write(str(epoch)+"\t"+str(round(temp_miou,2))+"\t")
                if self.name_classes is not None:
                    for ind_class in range(self.num_classes):
                        f.write(str(round(IoUs[ind_class] * 100, 2))+"\t")
                f.write("\n")
            # shutil.copy(os.path.join(self.log_dir, "epoch_miou.txt"),os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"))
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
##多线程方案，暂时搁置
                              
    def on_epoch_end_multithread(self, epoch, model_eval,thread_num):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")##改
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            thread_num = min(thread_num,20)#最多开20个线程
            files = [[] for _ in range(thread_num)]
            interval_num=len(self.image_ids)/thread_num
            count=0
            id=0
            for line in tqdm(self.image_ids):
                if line.strip()=="":
                    continue
                files[id%thread_num].append(line.strip())
                count+=1
                if count>interval_num:
                    count=0
                    id+=1
            pool=multiprocessing.Pool(thread_num)
            result=[]
            for i in range(thread_num):
                result.append(pool.apply_async(func=process,args=(self,files[i],pred_dir,)))
            pool.close()
            pool.join()        
                
            # for image_id in tqdm(self.image_ids):
            #     #-------------------------------#
            #     #   从文件中读取图像
            #     #-------------------------------#
            #     # image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")##改
            #     image_path  = image_id##改
            #     image       = Image.open(image_path)
            #     #------------------------------#
            #     #   获得预测txt
            #     #------------------------------#
            #     image       = self.get_miou_png(image)
            #     # image.save(os.path.join(pred_dir, image_id + ".png"))##改
            #     image.save(os.path.join(pred_dir,image_id.rsplit('/',1)[1].replace("jpg","png")))##改
                        
            print("Calculate miou.")
            _, IoUs, PA_Recall, Precision,name_classes = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, self.name_classes)##改  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            
            with open(os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"), 'a') as f:
                f.write(str(epoch)+"\t"+str(round(temp_miou,2))+"\t")
                if name_classes is not None:
                    for ind_class in range(self.num_classes):
                        f.write(str(round(IoUs[ind_class] * 100, 2))+"\t")
                f.write("\n")
            # shutil.copy(os.path.join(self.log_dir, "epoch_miou.txt"),os.path.join(self.log_dir.rsplit('/',1)[0], "Release/iou.txt"))
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)

def process(object,files,pred_dir):
    res=[]
    for image_id in tqdm(files):
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        # image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")##改
        image_path  = image_id##改
        image       = Image.open(image_path)
        #------------------------------#
        #   获得预测txt
        #------------------------------#
        image       = object.get_miou_png(image)
        # image.save(os.path.join(pred_dir, image_id + ".png"))##改
        image.save(os.path.join(pred_dir,image_id.rsplit('/',1)[1].replace("jpg","png")))##改
    
    return res

def cal_iou_multithread(preds_thread,img_thread,num_classes,name_classes):
    #计算miou
    hist = np.zeros((num_classes, num_classes))
    for j in tqdm(range(len(img_thread))):
        png,_=pre_process.generateMask_item(img_thread[j],num_classes,name_classes)        
        label=np.array(png)#标签
        pred=np.array(preds_thread[j])
        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()),img_thread[j]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)     
    return hist