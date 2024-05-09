import os
import random
import shutil
import base64
import json
import os.path as osp
from PIL import Image
# from labelme import utils
from skimage import img_as_ubyte
import cv2
import numpy as np
import uuid
import PIL
import math
from tqdm import tqdm
from PIL import Image, ImageDraw

#划分数据集为训练集和测试集，输入数据集txt文件路径和结果保存路径，返回train_txt，val_txt保存路径
def splitData(data_set_txt,ratio,save_dir):
    if not os.path.exists(data_set_txt) :
        print("{} doesn't exist!".format(data_set_txt))
        return "","",False
    if ratio<0 :
        print("ratio must exceed zero!,now ratio's value is {} !".format(ratio))
        return "","",False
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data=[]
    with open(data_set_txt,'r') as f:
        for line in f:
            data.append(line.strip())
    random.shuffle(data)
    total=len(data)
    offset=int(total*ratio)
    train=data[:offset]
    val=data[offset:]
    train_txt=os.path.join(save_dir,'train.txt')
    val_txt=os.path.join(save_dir,'val.txt')
    with open(train_txt,'w') as f:
        for tmp in train:
            f.write(tmp+'\n')
    with open(val_txt,'w') as f:
        for tmp in val:
            f.write(tmp+'\n')
    return train_txt,val_txt,True

#根据图片的-seg.json文件生成对应的mask图片
def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    # draw = PIL.ImageDraw.Draw(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask
def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        if label is None:
            continue
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shapeType", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins
def generateMask(data_set_txt,num_classes,save_dir,consider_type):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #json->mask
    processed_data_set_txt=os.path.join(save_dir,"processed_data.txt")
    exclude_data_set_txt=os.path.join(save_dir,"exclude_data.txt")
    processed_file=open(processed_data_set_txt,'w')
    exclude_file=open(exclude_data_set_txt,'w')
    with open(data_set_txt,'r') as f:
        for line in f:
            json_path=line.strip().rsplit('.',1)[0]+'-seg.json'
            if not os.path.exists(json_path):
                exclude_file.write(line.strip()+"\n")
                continue
            try:#尝试根据json文件生成对应的mask图片
                data=json.load(open(json_path))
                if data['polygons']==None:
                    data['polygons']=[]
                if num_classes==2:
                    for i in range(len(data['polygons'])):#类别如果为2，则将雾与烟合并
                        if data['polygons'][i]["label"]=="Wu":
                            data['polygons'][i]["label"]="Sm"
                for i in range(len(data['polygons'])):
                    if data['polygons'][i]["label"] ==None:
                        i+=1
                        continue
                    if data['polygons'][i]["label"] not in consider_type:
                        data['polygons'][i]["label"]="_background_"
                # img=cv2.imread(line.strip())#后续这里可以优化，直接读json文件中的imageHeight,imageWidth来获取图片高宽，shape(高，宽，通道数)，shape是个tuple
                # shape=[]
                # img=cv2.imread(line.strip())#后续这里可以优化，直接读json文件中的imageHeight,imageWidth来获取图片高宽，shape(高，宽，通道数)，shape是个tuple
                shape=[int(data['imageHeight']),int(data['imageWidth']),3]
                label_name_to_value = {'_background_': 0}
                name_classes=["_background_"]
                for type in consider_type:
                    label_value = len(label_name_to_value)
                    label_name_to_value[type]=label_value
                    name_classes.append(type)
                # for shape in sorted(data['polygons'], key=lambda x: x['label']):
                #     label_name = shape['label']
                #     if label_name==None:

                #         continue
                #     if label_name in label_name_to_value:
                #         label_value = label_name_to_value[label_name]
                #     else:
                #         label_value = len(label_name_to_value)
                #         label_name_to_value[label_name] = label_value

                # lbl, _ = utils.shapes_to_label(img.shape, data['polygons'], label_name_to_value)
                # lbl, _ = shapes_to_label(img.shape, data['polygons'], label_name_to_value)
                # lbl, _ = shapes_to_label(img.shape, data['polygons'], label_name_to_value)
                lbl, _ = shapes_to_label(shape, data['polygons'], label_name_to_value)
                mask_dst = img_as_ubyte(lbl)
                out_img = Image.fromarray(np.uint8(mask_dst))
                if not os.path.exists(line.strip().replace("jpg","png")):
                    out_img.save(line.strip().replace("jpg","png"))
            except:#代表json_path文件或者不符合json格式，或者为空,生成mask失败
                exclude_file.write(line.strip()+"\n")
            else:
                processed_file.write(line.strip()+"\n")

    processed_file.close()
    exclude_file.close()
    return processed_data_set_txt,exclude_data_set_txt,name_classes,True

def verify_json(data_set_txt,save_dir,num_classes,consider_type):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #json->mask
    processed_data_set_txt=os.path.join(save_dir,"processed_data.txt")
    exclude_data_set_txt=os.path.join(save_dir,"exclude_data.txt")
    processed_file=open(processed_data_set_txt,'w')
    exclude_file=open(exclude_data_set_txt,'w')
    with open(data_set_txt,'r') as f:
        for line in tqdm(f):
            json_path=line.strip().rsplit('.',1)[0]+'-seg.json'
            if not os.path.exists(json_path):
                exclude_file.write(line.strip()+"\n")
                continue
            try:#尝试根据json文件生成对应的mask图片
                data=json.load(open(json_path))
                if data['polygons']==None:
                    data['polygons']=[]
                if num_classes==2:
                    for i in range(len(data['polygons'])):#类别如果为2，则将雾与烟合并
                        if data['polygons'][i]["label"]=="Wu":
                            data['polygons'][i]["label"]="Sm"
                for i in range(len(data['polygons'])):
                    if data['polygons'][i]["label"] ==None:
                        i+=1
                        continue
                    if data['polygons'][i]["label"] not in consider_type:
                        data['polygons'][i]["label"]="_background_"
                # img=cv2.imread(line.strip())#后续这里可以优化，直接读json文件中的imageHeight,imageWidth来获取图片高宽，shape(高，宽，通道数)，shape是个tuple
                shape=[int(data['imageHeight']),int(data['imageWidth']),3]
                label_name_to_value = {'_background_': 0}
                name_classes=["_background_"]
                for type in consider_type:
                    label_value = len(label_name_to_value)
                    label_name_to_value[type]=label_value
                    name_classes.append(type)
                # lbl, _ = shapes_to_label(img.shape, data['polygons'], label_name_to_value)
                lbl, _ = shapes_to_label(shape, data['polygons'], label_name_to_value)
                mask_dst = img_as_ubyte(lbl)
                out_img = Image.fromarray(np.uint8(mask_dst))
            except:#代表json_path文件或者不符合json格式，或者为空,生成mask失败
                exclude_file.write(line.strip()+"\n")
            else:
                processed_file.write(line.strip()+"\n")
    name_classes=["_background_"]
    for type in consider_type:
        name_classes.append(type)
    processed_file.close()
    exclude_file.close()
    return processed_data_set_txt,exclude_data_set_txt,name_classes,True

def generateMask_item(jpg_name,num_classes,consider_type):
    line=jpg_name
    json_path=line.strip().rsplit('.',1)[0]+'-seg.json'
    #尝试根据json文件生成对应的mask图片
    data=json.load(open(json_path))
    if data['polygons']==None:
        data['polygons']=[]
    if num_classes==2:
        for i in range(len(data['polygons'])):#类别如果为2，则将雾与烟合并
            if data['polygons'][i]["label"]=="Wu":
                data['polygons'][i]["label"]="Sm"
    for i in range(len(data['polygons'])):
        if data['polygons'][i]["label"] ==None:
            i+=1
            continue
        if data['polygons'][i]["label"] not in consider_type:
            data['polygons'][i]["label"]="_background_"
    # img=cv2.imread(line.strip())#后续这里可以优化，直接读json文件中的imageHeight,imageWidth来获取图片高宽，shape(高，宽，通道数)，shape是个tuple
    # shape=[]
    # img=cv2.imread(line.strip())#后续这里可以优化，直接读json文件中的imageHeight,imageWidth来获取图片高宽，shape(高，宽，通道数)，shape是个tuple
    shape=[int(data['imageHeight']),int(data['imageWidth']),3]
    label_name_to_value = {'_background_': 0}
    for type in consider_type:
        if type =='_background_':
            continue
        label_value = len(label_name_to_value)
        label_name_to_value[type]=label_value
    # for shape in sorted(data['polygons'], key=lambda x: x['label']):
    #     label_name = shape['label']
    #     if label_name==None:
    #         continue
    #     if label_name in label_name_to_value:
    #         label_value = label_name_to_value[label_name]
    #     else:
    #         label_value = len(label_name_to_value)
    #         label_name_to_value[label_name] = label_value
    # lbl, _ = utils.shapes_to_label(img.shape, data['polygons'], label_name_to_value)
    # lbl, _ = shapes_to_label(img.shape, data['polygons'], label_name_to_value)
    # lbl, _ = shapes_to_label(img.shape, data['polygons'], label_name_to_value)
    lbl, _ = shapes_to_label(shape, data['polygons'], label_name_to_value)
    mask_dst = img_as_ubyte(lbl)
    out_img = Image.fromarray(np.uint8(mask_dst))
    # if not os.path.exists(line.strip().replace("jpg","png")):
    # out_img.save(line.strip().replace("jpg","png"))

    return out_img,True

from utils.utils import cvtColor, preprocess_input, resize_image, show_config
import copy
def generateMask_gt(jpg_name,num_classes,consider_type):
    image=Image.open(jpg_name)
    # if not os.path.exists(line.strip().replace("jpg","png")):
    # out_img.save(line.strip().replace("jpg","png"))
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #---------------------------------------------------------#
    image       = cvtColor(image)
    #---------------------------------------------------#
    #   对输入图像进行一个备份，后面用于绘图
    #---------------------------------------------------#
    old_img     = copy.deepcopy(image)
    line=jpg_name
    json_path=line.strip().rsplit('.',1)[0]+'-seg.json'
    #尝试根据json文件生成对应的mask图片
    data=json.load(open(json_path))
    if data['polygons']==None:
        data['polygons']=[]
    if num_classes==2:
        for i in range(len(data['polygons'])):#类别如果为2，则将雾与烟合并
            if data['polygons'][i]["label"]=="Wu":
                data['polygons'][i]["label"]="Sm"
    for i in range(len(data['polygons'])):
        if data['polygons'][i]["label"] ==None:
            i+=1
            continue
        if data['polygons'][i]["label"] not in consider_type:
            data['polygons'][i]["label"]="_background_"
    # img=cv2.imread(line.strip())#后续这里可以优化，直接读json文件中的imageHeight,imageWidth来获取图片高宽，shape(高，宽，通道数)，shape是个tuple
    # shape=[]
    # img=cv2.imread(line.strip())#后续这里可以优化，直接读json文件中的imageHeight,imageWidth来获取图片高宽，shape(高，宽，通道数)，shape是个tuple
    shape=[int(data['imageHeight']),int(data['imageWidth']),3]
    label_name_to_value = {'_background_': 0}
    for type in consider_type:
        if type =='_background_':
            continue
        label_value = len(label_name_to_value)
        label_name_to_value[type]=label_value
    # for shape in sorted(data['polygons'], key=lambda x: x['label']):
    #     label_name = shape['label']
    #     if label_name==None:
    #         continue
    #     if label_name in label_name_to_value:
    #         label_value = label_name_to_value[label_name]
    #     else:
    #         label_value = len(label_name_to_value)
    #         label_name_to_value[label_name] = label_value
    # lbl, _ = utils.shapes_to_label(img.shape, data['polygons'], label_name_to_value)
    # lbl, _ = shapes_to_label(img.shape, data['polygons'], label_name_to_value)
    # lbl, _ = shapes_to_label(img.shape, data['polygons'], label_name_to_value)
    lbl, _ = shapes_to_label(shape, data['polygons'], label_name_to_value)
    mask_dst = img_as_ubyte(lbl)
    out_img = Image.fromarray(np.uint8(mask_dst)*255)
    out_img       = cvtColor(out_img)
    # image   = Image.blend(old_img, out_img, 0.3)
    # return image,True
    return out_img,True

def comb(img1, img2, style='horizontal'):

    # img1, img2 = Image.open(png1), Image.open(png2)
    # 统一图片尺寸，可以自定义设置（宽，高）
    img1 = img1.resize((1280, 720), Image.ANTIALIAS)
    img2 = img2.resize((1280, 720), Image.ANTIALIAS)
    size1, size2 = img1.size, img2.size
    if style == 'horizontal':
        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        # joint.save('horizontal.jpg')
        return joint
    elif style == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        # joint.save('vertical.jpg')
        return joint

from scipy import linalg
def RGB_PCA(img):
    nrows, ncolumns  = img.shape[0:2] # 获取图片的行数与列数
    imgArrT = np.array(img, dtype=np.int64).reshape(-1, 3) # 每行是一个像素点
    mean = np.mean(imgArrT, axis=0) # 求所有像素点的平均值
    CenterImg = imgArrT - mean # 将每个像素减去平均值，以中心化
    
    covar = np.matmul(np.transpose(CenterImg), CenterImg)
    evalues, evectors = linalg.eig(covar) # 求特征值及特征向量
    principle_xis = evectors[:, np.argmax(evalues)] # 特征值最大的轴是主轴
    principle_xis = principle_xis/np.sum(principle_xis) # 将主轴的元素和变为1
    principle_xis = principle_xis[:, np.newaxis] # 将主轴转换成（3,1）
    principle_img = np.matmul(imgArrT, principle_xis) # 投影到主轴
    principle_img = np.uint8(principle_img.reshape((nrows, ncolumns))) # 还原为主轴灰度图
    colorimg=cv2.cvtColor(principle_img,cv2.COLOR_GRAY2RGB)
    return colorimg

from threading import Thread
import multiprocessing
def verify_json_multithread(data_set_txt,save_dir,num_classes,consider_type,thread_num):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #json->mask
    processed_data_set_txt=os.path.join(save_dir,"processed_data.txt")
    exclude_data_set_txt=os.path.join(save_dir,"exclude_data.txt")
    processed_file=open(processed_data_set_txt,'w')
    exclude_file=open(exclude_data_set_txt,'w')
    processed_file.close()
    exclude_file.close()
    lines=[]
    with open(data_set_txt,'r') as f:
        lines=f.readlines()
    thread_num = min(thread_num,20)#最多开20个线程

    files = [[] for _ in range(thread_num)]
    interval_num=len(lines)/thread_num#均分文件数目
    count=0
    id=0
    for line in tqdm(lines):
        if line.strip()=="":
            continue
        files[id%thread_num].append(line.strip())
        count+=1
        if count>interval_num:
            count=0
            id+=1
    pool=multiprocessing.Pool(thread_num)
    for i in range(thread_num):
        pool.apply_async(func=process,args=(files[i],num_classes,consider_type,processed_data_set_txt,exclude_data_set_txt,),callback=setcallback)        
    
    name_classes=["_background_"]
    for type in consider_type:
        name_classes.append(type)

    pool.close()
    pool.join()
    return processed_data_set_txt,exclude_data_set_txt,name_classes,True

def setcallback(res):
    with open(res["processed_data_set_txt"],"a+") as f:
        for line in tqdm(res["processed_data"]):
            f.write(line)
    with open(res["exclude_data_set_txt"],"a+") as f:
        for line in tqdm(res["exclude_data"]):
            f.write(line)
def process(files,num_classes,consider_type,processed_data_set_txt,exclude_data_set_txt):
    processed_data=[]
    exclude_data=[]
    for line in tqdm(files):
            json_path=line.strip().rsplit('.',1)[0]+'-seg.json'
            if not os.path.exists(json_path):
                exclude_data.append(line.strip()+"\n")
                continue
            try:#尝试根据json文件生成对应的mask图片
                data=json.load(open(json_path))
                if data['polygons']==None:
                    data['polygons']=[]
                if num_classes==2:
                    for i in range(len(data['polygons'])):#类别如果为2，则将雾与烟合并
                        if data['polygons'][i]["label"]=="Wu":
                            data['polygons'][i]["label"]="Sm"
                for i in range(len(data['polygons'])):
                    if data['polygons'][i]["label"] ==None:
                        i+=1
                        continue
                    if data['polygons'][i]["label"] not in consider_type:
                        data['polygons'][i]["label"]="_background_"
                # img=cv2.imread(line.strip())#后续这里可以优化，直接读json文件中的imageHeight,imageWidth来获取图片高宽，shape(高，宽，通道数)，shape是个tuple
                shape=[int(data['imageHeight']),int(data['imageWidth']),3]
                label_name_to_value = {'_background_': 0}
                name_classes=["_background_"]
                for type in consider_type:
                    label_value = len(label_name_to_value)
                    label_name_to_value[type]=label_value
                    name_classes.append(type)
                # lbl, _ = shapes_to_label(img.shape, data['polygons'], label_name_to_value)
                lbl, _ = shapes_to_label(shape, data['polygons'], label_name_to_value)                
                mask_dst = img_as_ubyte(lbl)
                out_img = Image.fromarray(np.uint8(mask_dst))
            except:#代表json_path文件或者不符合json格式，或者为空,生成mask失败
                exclude_data.append(line.strip()+"\n")
            else:
                processed_data.append(line.strip()+"\n")
    res={}
    res["processed_data_set_txt"]=processed_data_set_txt
    res["processed_data"]=processed_data
    res["exclude_data_set_txt"]=exclude_data_set_txt
    res["exclude_data"]=exclude_data
    return res
