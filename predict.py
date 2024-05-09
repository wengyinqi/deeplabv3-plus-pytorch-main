#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import os
import time

import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3
import argparse
class predict_args:
    def __init__(self):
        parser=argparse.ArgumentParser(description='输入参数')
        parser.add_argument('--data_set_txt','-ds',type=str,required=False,help='diff,数据集txt文件路径,包含图片的完整路径名，为jpg格式')
        parser.add_argument('--save_dir','-sd',type=str,required=False,help='diff,结果输出目录')     
        parser.add_argument('--model_path','-mp', type=str, required=False,default="model_data/deeplab_mobilenetv2.pth", help="选择使用的模型")        
        parser.add_argument('--train_ratio','-tr',type=float,required=False,default=0,help='diff,训练集占比')        
        parser.add_argument('--num_classes','-nc', type=int, required=False,default=2, help="diff,自己需要的分类个数")
        parser.add_argument('--consider_type','-ct',nargs='+',required=False,default=['Sm'],help="diff,都会加有背景类_background，此处只加入需要考虑的分割类，默认有Sm")
        parser.add_argument('--thread_num','-tn', type=int, required=False,default=20, help="diff,线程数")  
        parser.add_argument('--mode','-mo', type=str, required=False,default="predict", help="预测模式，默认为预测单张图片")                      
        parser.add_argument('--pt_path','-pp', type=str, required=False,default="", help="预测模式，默认为预测单张图片")                      
        parser.add_argument('--onnx_save_path','-os', type=str, required=False,default="", help="export_onnx有效，预测模式，默认为预测单张图片")                      
        parser.add_argument('--oriImg_path','-op', type=str, required=False,default="", help="plt_diff有效，原图路径，精确到pic文件夹上一级")                      
        parser.add_argument('--wzcImg_path','-wp', type=str, required=False,default="", help="plt_diff有效，文志川处理后的数据保存路径，精确到pic文件夹上一级")                          
        parser.add_argument('--diff','-di',nargs='+',required=False,default=['D2','D3','D4','D5'],help="plt_diff有效，帧差类别")
        parser.add_argument('--dir_origin_path','-do', type=str, required=False,default="", help="dir_predict或dir_predict_diff有效，精确到彩色原图或帧差图上一级的原图路径")                          
        parser.add_argument('--dir_save_path','-dp', type=str, required=False,default="", help="dir_predict或dir_predict_diff有效，精确到图片上一级的原图路径")                          
        parser.add_argument('--ori_image_path_dir','-oi', type=str, required=False,default="", help="dir_predict_diff有效，精确到彩色原图上一级的原图路径")                          
        parser.add_argument('--count',type=bool,required=False,default=False,help='，predict模式有效，指定了是否进行目标的像素点计数（即面积）与比例计算，默认为False')
        parser.add_argument('--name_classes','-nac',nargs='+',required=False,default=["background","dust"],help="predict模式有效，为类别名")
        parser.add_argument('--video_path','-vp', type=str, required=False,default="", help="video有效，需要测试的视频文件的路径")                          
        parser.add_argument('--video_save_path','-vs', type=str, required=False,default="", help="video有效，测试得到的结果视频的保存路径")                          
        parser.add_argument('--video_path_dir','-vpd', type=str, required=False,default="", help="video_dir有效，视频文件夹，精确到测试的多个视频文件的上一级")                          
        parser.add_argument('--video_save_path_dir','-vsd', type=str, required=False,default="", help="video_dir有效，保存视频的文件夹，精确到保存视频文件的上一级")                          
        self.parameters= parser.parse_args()
'''
python predict.py --data_set_txt=100.txt --save_dir=/home/sma/data960/WYQ/PltData/NanTongZhongTian --train_ratio=0 --model_path=/home/wyq/deeplabv3-plus-pytorch-main/ntzt_D/loss_2023_10_28_21_33_57_Sm_80.04/best.pth
'''
if __name__ == "__main__":
    parameters=predict_args().parameters
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    deeplab = DeeplabV3(model_path=parameters.model_path,mix_type=0)
    # deeplab = DeeplabV3()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    # mode ='plt_diff'#'export_onnx'#'test_diff_img'#'test_single_img'# 'dir_predict'#'export_onnx'#'video'#'dir_predict'#"predict"
    mode=parameters.mode
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    # count           = False
    # name_classes    = ["background","dust", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    count=parameters.count
    name_classes=parameters.name_classes
    # name_classes    = ["background","cat","dog"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    # video_path_      = 'D:\\WYQ\\DeeplabV3\\全彩原始视频'#0
    # video_save_path_ = 'video_out'#""
    video_path=parameters.video_path
    video_save_path=parameters.video_save_path
    video_path_dir=parameters.video_path_dir
    video_save_path_dir=parameters.video_save_path_dir
    # video_path      = 'D:\\WYQ\\DeeplabV3\\Deeplabv3Plus_Experiment\\Deeplabv3Plus\\videos\\ntzt.mp4'#0
    # video_save_path = 'D:\\WYQ\\DeeplabV3\\Deeplabv3Plus_Experiment\\Deeplabv3Plus\\videos\\ntzt_S_72.mp4'#""
    video_fps       = 10#25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    # dir_origin_path =    "img/pic/pic"
    # dir_save_path   = "img_out/ori_S"
    dir_origin_path=parameters.dir_origin_path
    dir_save_path=parameters.dir_save_path
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    # pt_path="/home/wyq/deeplabv3-plus-pytorch-main/ntzt_D/loss_2023_10_28_21_33_57_Sm_80.04/best.pth"
    # onnx_save_path  = "model_data/ntzt_D_Sm_80.04.onnx"
    pt_path=parameters.pt_path
    onnx_save_path=parameters.onnx_save_path

    if mode == "predict":
        '''
        python predict.py --mode=predict --model_path=single.pth
        '''
        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                # r_image,_ = deeplab.detect_image(image, count=count, name_classes=name_classes)
                # r_image.show()
                # r_image.save('img/result.jpg')
                mask=np.asarray(deeplab.get_miou_png(image))
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                drawimg=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                cv2.drawContours(drawimg,contours,-1,(0,0,255),3)
                # cv2.imshow("result",drawimg)
                cv2.imwrite("img/result.jpg",drawimg)
                

    elif mode == "video":
        '''
        python predict.py --mode=video --model_path=single.pth --video_path=test.mp4 --video_save_path=test_deeplabv3plus.mp4
        '''
        # for video_name in os.listdir(video_path_):
        #     video_path=os.path.join(video_path_,video_name)
        #     video_save_path=os.path.join(video_save_path_,video_name.split('.')[0]+'_deeplabv3.mp4')
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_fps=int(capture.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # # 格式转变，BGRtoRGB
            # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # # 转变成Image
            # frame = Image.fromarray(np.uint8(frame))
            # # 进行检测
            # frame= np.array(deeplab.detect_image(frame)[0])
            # # RGBtoBGR满足opencv显示格式
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            mask=np.asarray(deeplab.get_miou_png(Image.fromarray(np.uint8(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame,contours,-1,(0,0,255),3)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # cv2.imshow("video",frame)
            # c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)
            # if c==27:
            #     capture.release()
            #     break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        # cv2.destroyAllWindows()
    elif mode == "video_dir":
        '''
        python predict.py --mode=video_dir --model_path=single.pth --video_path_dir=videos --video_save_path_dir=videos_out
        '''
        if not os.path.exists(video_save_path_dir):
            os.makedirs(video_save_path_dir)
        for video_name in os.listdir(video_path_dir):
            video_path=os.path.join(video_path_dir,video_name)
            video_save_path=os.path.join(video_save_path_dir,video_name.split('.')[0]+'_deeplabv3.mp4')
            capture=cv2.VideoCapture(video_path)
            if video_save_path!="":
                # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                video_fps=int(capture.get(cv2.CAP_PROP_FPS))
                out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
            ref, frame = capture.read()
            if not ref:
                raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
            fps = 0.0
            while(True):
                t1 = time.time()
                # 读取某一帧
                ref, frame = capture.read()
                if not ref:
                    break
                # # 格式转变，BGRtoRGB
                # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # # 转变成Image
                # frame = Image.fromarray(np.uint8(frame))
                # # 进行检测
                # frame= np.array(deeplab.detect_image(frame)[0])
                # # RGBtoBGR满足opencv显示格式
                # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                
                mask=np.asarray(deeplab.get_miou_png(Image.fromarray(np.uint8(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))))
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame,contours,-1,(0,0,255),3)
                
                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                print("fps= %.2f"%(fps))
                frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # cv2.imshow("video",frame)
                # c= cv2.waitKey(1) & 0xff 
                if video_save_path!="":
                    out.write(frame)
                # if c==27:
                #     capture.release()
                #     break
            print("Video Detection Done!")
            capture.release()
            if video_save_path!="":
                print("Save processed video to the path :" + video_save_path)
                out.release()
            # cv2.destroyAllWindows()
    
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":#使用单帧模型预测单帧图，并画在原图上
        '''
        python predict.py --mode=dir_predict --model_path=single.pth --dir_origin_path=img/pic --dir_save_path=img_out/pic
        其中，dir_origin_path为彩色原图文件夹，精确到图片上一级，dir_save_path为画在原图后的图片保存路径，精确到图片上一级
        '''
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                # r_image,_     = deeplab.detect_image(image)
                # if not os.path.exists(dir_save_path):
                #     os.makedirs(dir_save_path)
                # r_image.save(os.path.join(dir_save_path, img_name))
                mask=np.asarray(deeplab.get_miou_png(image))
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                drawimg=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                cv2.drawContours(drawimg,contours,-1,(0,0,255),3)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                cv2.imwrite(os.path.join(dir_save_path,img_name),drawimg)
    elif mode == "dir_predict_diff":#"dir_predict_ori_mask":#使用帧差模型预测帧差图，并画在原图上
        '''
        python predict.py --mode=dir_predict_diff --model_path=single.pth --dir_origin_path=img/pic --dir_save_path=img_out/pic ori_image_path_dir=ori_img/pic
        其中，dir_origin_path为帧差图文件夹，精确到图片上一级，dir_save_path为画在原图后的图片保存路径，精确到图片上一级，ori_image_path_dir为制作帧差的原图保存路径，精确到图片上一级
        '''
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        # ori_image_path_dir="img/pic/pic"
        ori_image_path_dir=parameters.ori_image_path_dir
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                # r_image,mask     = deeplab.detect_image(image)
                # ori_image=Image.open(os.path.join(ori_image_path_dir,img_name.split("_")[0]+".jpg"))
                # ori_mask=Image.blend(ori_image, mask, 0.3)
                # if not os.path.exists(dir_save_path):
                #     os.makedirs(dir_save_path)
                # ori_mask.save(os.path.join(dir_save_path, img_name))     
                mask=np.asarray(deeplab.get_miou_png(image))
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                ori_image=cv2.imread(os.path.join(ori_image_path_dir,img_name.split("_")[0]+".jpg"))
                cv2.drawContours(ori_image,contours,-1,(0,0,255),3)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                cv2.imwrite(os.path.join(dir_save_path,img_name),ori_image)
    elif mode == "export_onnx":
        '''
        python predict.py --mode=export_onnx --model_path=/home/wyq/deeplabv3-plus-pytorch-main/ntzt_D/loss_2023_10_28_21_33_57_Sm_80.04/best.pth --pt_path=/home/wyq/deeplabv3-plus-pytorch-main/ntzt_D/loss_2023_10_28_21_33_57_Sm_80.04/best.pth --onnx_save_path=model_data/ntzt_D_Sm_80.04.onnx
        '''
        # deeplab.convert_to_onnx(simplify, onnx_save_path)
        deeplab.convert_to_onnx(simplify,pt_path, onnx_save_path)
    elif mode=="test_single_img":
        import os
        from tqdm import tqdm
        import pre_process
        f=open("DuGang-S/val.txt")
        # img_names = os.listdir(dir_origin_path)
        # for img_name in tqdm(img_names):
        for line in tqdm(f):
            image_path=line.strip()
            img_name=image_path.rsplit("/",1)[1]
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                # image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image,_     = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                # r_image.save(os.path.join(dir_save_path,"val/pre", img_name))  
                gt_img,_=pre_process.generateMask_gt(image_path,2,['Sm'])
                # gt_img.save(os.path.join(dir_save_path,"val/gt",img_name.replace(".jpg",".png")))
                image   = Image.blend(r_image, gt_img, 0.3)
                image.save(os.path.join(dir_save_path,"mix", img_name))
        f.close()
    elif mode=='test_diff_img':
        import os
        from tqdm import tqdm
        import pre_process
        from utils.utils import cvtColor, preprocess_input, resize_image, show_config
        from opts import args
        f=open("/home/wyq/Deeplabv3Data/ZhongTian/data_txt/static_single.txt")
        single_file=[]
        # single_path=[]
        for line in tqdm(f):
            line=line.strip()
            # single_path.append(line)
            # file=line.split("/",8)[-1]
            single_file.append(line)
        f.close()
        parameters=args().parameters
        processed_data_set_txt,exclude_data_set_txt,name_classes,flag=pre_process.verify_json(parameters.data_set_txt,parameters.save_dir,parameters.num_classes,parameters.consider_type)
        train_txt,val_txt,flag=pre_process.splitData(processed_data_set_txt,parameters.train_ratio,parameters.save_dir)
        dd=open(val_txt)
        # dd=open("/home/wyq/Deeplabv3Data/ZhongTian/data_txt/DD_data.txt")
        for line in tqdm(dd):
            print(line)
            image_path=line.strip()
            img_name=image_path.rsplit("/",1)[1]
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                # image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image ,mask    = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                gt_img,_=pre_process.generateMask_gt(image_path,2,['Sm'])
                image   = Image.blend(r_image, gt_img, 0.3)#帧差图上画
                diff_img=image_path.rsplit("/",1)
                img1=os.path.join(diff_img[0],diff_img[1].split("_")[0]+"-00.jpg").replace("FeiGangLiaoPeng-DD/FeiGangLiaoPeng-DD","FeiGangLiaoPeng-S")
                img2=os.path.join(diff_img[0],diff_img[1].split("_")[1]).replace("FeiGangLiaoPeng-DD/FeiGangLiaoPeng-DD","FeiGangLiaoPeng-S")
                if img1 in single_file:
                    ori=Image.open(img1)
                    ori = cvtColor(ori)
                    img=Image.blend(ori,mask,0.3)
                    img=Image.blend(img,gt_img,0.3)
                    res=pre_process.comb(image,img)
                else:
                    ori=Image.open(img2)
                    ori = cvtColor(ori)
                    img=Image.blend(ori,mask,0.3)
                    img=Image.blend(img,gt_img,0.3)
                    res=pre_process.comb(image,img)
                res.save(os.path.join(dir_save_path,"diff",image_path.split("/",9)[-1].replace("/","-")))
        dd.close()
    elif mode=='plt_diff':
        '''
        python predict.py --data_set_txt=/home/sma/data960/WYQ/DataTxt/ntzt-ztt/ntzt-ztt_D5.txt --save_dir=/home/sma/data960/WYQ/PltData/NanTongZhongTian-ztt --model_path=/home/wyq/deeplabv3-plus-pytorch-main/ntzt_D/loss_2023_10_28_21_33_57_Sm_80.04/best.pth --train_ratio=0 --mode=plt_diff --oriImg_path=/home/sma/datatmp2/ztt-savepic/NanTongZhongTian --wzcImg_path=/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt
        '''
        import os
        from tqdm import tqdm
        import pre_process
        from utils.utils import cvtColor, preprocess_input, resize_image, show_config
        # from opts import args
        # plt_diff_path="/home/sma/data960/WYQ/PltData"
        # parameters=args().parameters
        # processed_data_set_txt,exclude_data_set_txt,name_classes,flag=pre_process.verify_json_multithread(parameters.data_set_txt,parameters.save_dir,parameters.num_classes,parameters.consider_type,20)
        processed_data_set_txt,exclude_data_set_txt,name_classes,flag=pre_process.verify_json_multithread(parameters.data_set_txt,parameters.save_dir,parameters.num_classes,parameters.consider_type,parameters.thread_num)
        if not flag:
            print("generate mask error!")
            exit(1)
        train_txt,val_txt,flag=pre_process.splitData(processed_data_set_txt,parameters.train_ratio,parameters.save_dir)
        #val_txt为需要画图的文件路径txt
        #先画
        oriImg_path=parameters.oriImg_path
        wzcImg_path=parameters.wzcImg_path
        diff=parameters.diff
        # oriImg_path="/home/sma/datatmp2/ztt-savepic/NanTongZhongTian"#原图路径,精确到pic上一级
        # wzcImg_path="/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt"#文处理后的数据保存路径（裁剪帧差后的数据）,精确到pic上一级
        # oriImg_path="/home/sma/data/NanTongZhongTian"#原图路径
        # wzcImg_path="/home/sma/data960/WYQ/WZCData/NanTongZhongTian"#文处理后的数据保存路径（裁剪帧差后的数据）
        # flag_plt={}
        # diff=["D2","D3","D4","D5"]#帧差间隔类别
        with open(val_txt,'r') as f:
            for line in tqdm(f):
                #将画的图与原图上下拼接，画图在上，原图在下
                line=line.strip()#预测图烟羽区域用红色
                line_dir=""#画图保存路径
                first_img_path=""#第一张原图路径，烟羽区域用黄色
                second_img_path=""#第二张原图路径，烟羽区域用绿色
                save_img_path=""#结果图路径
                point={"x":0,"y":0}#裁剪的左上点坐标
                # plted=False#是否已经画了，默认没画
                number=0#裁剪图片的序号，要画在图上，用黑色
                for Dx in diff:#判断line属于哪个帧差间隔类别Dx
                    if Dx in line:
                        min_dir=line.split("-"+Dx+"/")[1].rsplit("/",1)[0]
                        if not os.path.exists(os.path.join(parameters.save_dir,Dx,min_dir)):
                            os.makedirs(os.path.join(parameters.save_dir,Dx,min_dir))
                        line_dir=os.path.join(parameters.save_dir,Dx,min_dir)
                        first_img_path=line.replace(wzcImg_path,oriImg_path).replace("-"+Dx,"").rsplit("_",4)[0]+".jpg"
                        second_img_path=first_img_path.rsplit("/",1)[0]+"/"+line.rsplit("_",4)[1]+".jpg"
                        number=int(line.rsplit("_",3)[1])
                        point["x"]=int(line.rsplit("_",3)[2])
                        point["y"]=int(line.rsplit("_",3)[3].split(".")[0])
                        save_img_path=os.path.join(parameters.save_dir,Dx,min_dir,line.rsplit("/",1)[1].rsplit("_",3)[0]+".jpg")
                        # if line.rsplit("_",3)[0] not in flag_plt:
                        #     flag_plt[line.rsplit("_",3)[0]]=True
                        # else:
                        #     plted=True
                        break
                # if not plted:#未画图
                if not os.path.exists(save_img_path):#save_img_path还不存在，代表还未画图，需要读取第一张原图和第二章原图
                    first_img=cv2.imread(first_img_path)#opencv格式
                    plted_img=first_img.copy()
                    first_img_mask=np.asarray(pre_process.generateMask_item(first_img_path,2,['Sm'])[0])#opencv格式,二值图
                    second_img_mask=np.asarray(pre_process.generateMask_item(second_img_path,2,['Sm'])[0])#opencv格式，二值图
                    pred_img_mask=np.asarray(deeplab.get_miou_png(Image.open(line)))#opencv格式，二值图
                    first_contours,_=cv2.findContours(first_img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(plted_img,first_contours,-1,(255,0,0),7)#第一张蓝
                    second_contours,_=cv2.findContours(second_img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(plted_img,second_contours,-1,(0,255,0),5)#第二张绿
                    pred_contours,_=cv2.findContours(pred_img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(plted_img,pred_contours,-1,(0,0,255),3,offset=(point["x"],point["y"]))#预测区域红
                    res = np.vstack((plted_img, first_img))#纵向拼接
                    # res = np.concatenate([plted_img, first_img], axis=1)#水平拼接
                    cv2.imwrite(save_img_path,res)
                else:#save_img_path已经存在，代表已经画了一个裁剪预测图，直接读取save_img_path，在此基础上画下一个裁剪预测图
                    plted_img=cv2.imread(save_img_path)
                    pred_img_mask=np.asarray(deeplab.get_miou_png(Image.open(line)))#opencv格式，二值图
                    pred_contours,_=cv2.findContours(pred_img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(plted_img,pred_contours,-1,(0,number*10%255,255),3,offset=(point["x"],point["y"]))#预测区域红
                    cv2.imwrite(save_img_path,plted_img)
    elif mode == 'plt_single_nojson':
        '''
        python predict.py --data_set_txt=/home/sma/data960/WYQ/DataTxt/ntzt-ztt/ntzt-ztt_D5.txt --save_dir=/home/sma/data960/WYQ/PltData/NanTongZhongTian-ztt --model_path=/home/wyq/deeplabv3-plus-pytorch-main/ntzt_D/loss_2023_10_28_21_33_57_Sm_80.04/best.pth --mode=plt_diff_nojson --oriImg_path=/home/sma/datatmp2/ztt-savepic/NanTongZhongTian --wzcImg_path=/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt
        '''
        import os
        from tqdm import tqdm
        import pre_process
        from utils.utils import cvtColor, preprocess_input, resize_image, show_config
        oriImg_path=parameters.oriImg_path
        wzcImg_path=parameters.wzcImg_path
        val_txt=parameters.data_set_txt
        diff=['D0']
        with open(val_txt,'r') as f:
            for line in tqdm(f):
                line=line.strip()
                first_img_path=""
                save_img_path=""
                point={"x":0,"y":0}
                number=0
                for Dx in diff:
                    if Dx in line:
                        save_img_dir=line.replace(wzcImg_path,parameters.save_dir).rsplit("/",1)[0]
                        # min_dir=line.split("-"+Dx+"/")[1].rsplit("/",1)[0]
                        # if not os.path.exists(os.path.join(parameters.save_dir,Dx,min_dir)):
                        #     os.makedirs(os.path.join(parameters.save_dir,Dx,min_dir))
                        if not os.path.exists(save_img_dir):
                            os.makedirs(os.path.join(save_img_dir))
                        first_img_path=line.replace(wzcImg_path,oriImg_path).replace("-"+Dx,"").rsplit("_",3)[0]+".jpg"
                        number=int(line.rsplit("_",3)[1])
                        point["x"]=int(line.rsplit("_",3)[2])
                        point["y"]=int(line.rsplit("_",3)[3].split(".")[0])
                        # save_img_path=os.path.join(parameters.save_dir,Dx,min_dir,line.rsplit("/",1)[1].rsplit("_",3)[0]+".jpg")
                        save_img_path=os.path.join(save_img_dir,line.rsplit("/",1)[1].rsplit("_",3)[0]+".jpg")
                        # if line.rsplit("_",3)[0] not in flag_plt:
                        #     flag_plt[line.rsplit("_",3)[0]]=True
                        # else:
                        #     plted=True
                        break
                if not os.path.exists(save_img_path):#save_img_path还不存在，代表还未画图，需要读取第一张原图和第二章原图
                    first_img=cv2.imread(first_img_path)#opencv格式
                    plted_img=first_img.copy()
                    # first_img_mask=np.asarray(pre_process.generateMask_item(first_img_path,2,['Sm'])[0])#opencv格式,二值图
                    # second_img_mask=np.asarray(pre_process.generateMask_item(second_img_path,2,['Sm'])[0])#opencv格式，二值图
                    pred_img_mask=np.asarray(deeplab.get_miou_png(Image.open(line)))#opencv格式，二值图
                    # first_contours,_=cv2.findContours(first_img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    # cv2.drawContours(plted_img,first_contours,-1,(255,0,0),7)#第一张蓝
                    # second_contours,_=cv2.findContours(second_img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    # cv2.drawContours(plted_img,second_contours,-1,(0,255,0),5)#第二张绿
                    pred_contours,_=cv2.findContours(pred_img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(plted_img,pred_contours,-1,(0,0,255),3,offset=(point["x"],point["y"]))#预测区域红
                    res = np.vstack((plted_img, first_img))#纵向拼接
                    # res = np.concatenate([plted_img, first_img], axis=1)#水平拼接
                    cv2.imwrite(save_img_path,res)
                else:#save_img_path已经存在，代表已经画了一个裁剪预测图，直接读取save_img_path，在此基础上画下一个裁剪预测图
                    plted_img=cv2.imread(save_img_path)
                    pred_img_mask=np.asarray(deeplab.get_miou_png(Image.open(line)))#opencv格式，二值图
                    pred_contours,_=cv2.findContours(pred_img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(plted_img,pred_contours,-1,(0,number*10%255,255),3,offset=(point["x"],point["y"]))#预测区域红
                    cv2.imwrite(save_img_path,plted_img)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
