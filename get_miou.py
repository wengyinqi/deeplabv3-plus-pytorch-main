import os

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results,fast_hist,per_class_iu,per_class_PA_Recall,per_class_Precision

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
'''
运行命令：
python get_miou.py --data_set=/home/sma/data960/WYQ/DataTxt/ntzt-ztt/ntzt-ztt_D5.txt --save_dir=ntzt-ztt-D-testIou/D7 --train_ratio=0 --model_path=/home/wyq/deeplabv3-plus-pytorch-main/ntzt_D/loss_2023_10_28_21_33_57_Sm_80.04/best.pth
'''
# from opts import args
import pre_process
import numpy as np
import argparse
class get_miou_args:
    def __init__(self):
        parser=argparse.ArgumentParser(description='输入参数')
        parser.add_argument('--data_set_txt','-ds',type=str,required=True,help='数据集txt文件路径,包含图片的完整路径名，为jpg格式')
        parser.add_argument('--save_dir','-sd',type=str,required=True,help='结果输出目录')     
        parser.add_argument('--model_path','-mp', type=str, required=True,default="model_data/deeplab_mobilenetv2.pth", help="选择使用的模型")        
        parser.add_argument('--train_ratio','-tr',type=float,required=False,default=0,help='训练集占比')        
        parser.add_argument('--num_classes','-nc', type=int, required=False,default=2, help="自己需要的分类个数")
        parser.add_argument('--consider_type','-ct',nargs='+',required=False,default=['Sm'],help="都会加有背景类_background，此处只加入需要考虑的分割类，默认有Sm")
        parser.add_argument('--thread_num','-tn', type=int, required=False,default=20, help="线程数")                
        parser.add_argument('--miou_mode','-mm', type=int, required=False,default=3, help="miou计算模式，默认为3模式")                        
        self.parameters= parser.parse_args()
if __name__ == "__main__":
    parameters=get_miou_args().parameters
    # processed_data_set_txt,exclude_data_set_txt,name_classes,flag=pre_process.verify_json(parameters.data_set_txt,parameters.save_dir,parameters.num_classes,parameters.consider_type)
    # processed_data_set_txt,exclude_data_set_txt,name_classes,flag=pre_process.verify_json_multithread(parameters.data_set_txt,parameters.save_dir,parameters.num_classes,parameters.consider_type,20)#多线程
    processed_data_set_txt,exclude_data_set_txt,name_classes,flag=pre_process.verify_json_multithread(parameters.data_set_txt,parameters.save_dir,parameters.num_classes,parameters.consider_type,parameters.thread_num)#多线程
    if not flag:
       print("generate mask error!")
       exit(1)
    train_txt,val_txt,flag=pre_process.splitData(processed_data_set_txt,parameters.train_ratio,parameters.save_dir)
    with open(val_txt,"r") as f:##改
        val_lines = f.readlines()
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = parameters.miou_mode#3#0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = parameters.num_classes#2
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = name_classes#["background","dust", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    # image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    image_ids=val_lines
    image_ids = [image_id.split()[0] for image_id in image_ids]
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"#"evaluation"#"miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')#os.path.join(miou_out_path, 'val_rename')#os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image_path=image_id
            image       = Image.open(image_path)
            image       = deeplab.get_miou_png(image)
            # image.save(os.path.join(pred_dir, image_id + ".png"))
            image.save(os.path.join(pred_dir,image_id.rsplit('/',1)[1].replace("jpg","png")))##改
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision,name_classes = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        # show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        show_results(parameters.save_dir, hist, IoUs, PA_Recall, Precision, name_classes)
        
        temp_miou = np.nanmean(IoUs) * 100
        
        with open(os.path.join(parameters.save_dir, "iou.txt"), 'w') as f:
                f.write("miou\t")
                if name_classes is not None:
                    for ind_class in range(num_classes):
                        f.write(name_classes[ind_class]+"\t")
                f.write("\n")
        with open(os.path.join(parameters.save_dir, "iou.txt"), 'a') as f:
            f.write(str(round(temp_miou,2))+"\t")
            if name_classes is not None:
                for ind_class in range(num_classes):
                    f.write(str(round(IoUs[ind_class] * 100, 2))+"\t")
            f.write("\n")
    
    if miou_mode ==3:#代表不生成预测图，直接计算iou
        deeplab = DeeplabV3(model_path=parameters.model_path)
        hist = np.zeros((num_classes, num_classes))
        for image_id in tqdm(image_ids):
            # image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image_path=image_id
            pred_img       = deeplab.get_miou_png(Image.open(image_path))
            png,_=pre_process.generateMask_item(image_path,num_classes,name_classes)  
            label=np.array(png)
            pred=np.array(pred_img)       
            # 如果图像分割结果与标签的大小不一样，这张图片就不计算
            if len(label.flatten()) != len(pred.flatten()):  
                print(
                    'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}'.format(
                        len(label.flatten()), len(pred.flatten()),image_path))
                continue
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)           
        IoUs=per_class_iu(hist)
        PA_Recall   = per_class_PA_Recall(hist)
        Precision   = per_class_Precision(hist)
        hist=np.array(hist, np.int_)
        # show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        show_results(parameters.save_dir, hist, IoUs, PA_Recall, Precision, name_classes)
        
        temp_miou = np.nanmean(IoUs) * 100
        
        with open(os.path.join(parameters.save_dir, "iou.txt"), 'w') as f:
                f.write("miou\t")
                if name_classes is not None:
                    for ind_class in range(num_classes):
                        f.write(name_classes[ind_class]+"\t")
                f.write("\n")
        with open(os.path.join(parameters.save_dir, "iou.txt"), 'a') as f:
            f.write(str(round(temp_miou,2))+"\t")
            if name_classes is not None:
                for ind_class in range(num_classes):
                    f.write(str(round(IoUs[ind_class] * 100, 2))+"\t")
            f.write("\n")