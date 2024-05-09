import os
from tqdm import tqdm
import multiprocessing
import argparse
class args:
    def __init__(self):
        parser=argparse.ArgumentParser(description='输入参数')
        parser.add_argument('--dir','-d',nargs='+',required=True,default=[],type=str,help="目录列表")
        parser.add_argument('--txt','-t',type=str,required=True,help='生成的txt文件')     
        parser.add_argument('--mode','-m',type=str,required=False,default="Files",help='模式，Files or Folders')
        self.parameters= parser.parse_args()
'''
python generateTxt.py --mode=Files --dir /home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/DongFeiGangKu-D2/ /home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/FeiGangJiaGongZhongXin-D2/ --txt /home/sma/data960/WYQ/DataTxt/ntzt-ztt/ntzt-ztt_D2.txt
'''
if __name__=="__main__":
    parameters=args().parameters
    if not os.path.exists(parameters.txt.rsplit("/",1)[0]):
        os.makedirs(parameters.txt.rsplit("/",1)[0])
    if parameters.mode=="Files":
        with open(parameters.txt,'w') as f:
            for path in tqdm(parameters.dir):
                for root,dirs,files in os.walk(path):
                    for file in tqdm(files):
                        if 'jpg' in file:
                            f.write(os.path.join(root,file)+"\n")
    if parameters.mode=="Folders":
        folders={}
        with open(parameters.txt,'w') as f:
            for path in tqdm(parameters.dir):
                for root,dirs,files in os.walk(path):
                    for file in tqdm(files):
                        if 'jpg' in file:
                            if os.path.join(root,file).rsplit("/",1)[0] not in folders:
                                f.write(os.path.join(root,file).rsplit("/",1)[0]+"\n")
                                folders[os.path.join(root,file).rsplit("/",1)[0]]=True
                            
#ntzt
# f=open("/home/sma/data960/WYQ/DataTxt/ntzt-ztt/ntzt-ztt_D2.txt",'w')
# for root,dirs,files in os.walk("/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/DongFeiGangKu-D2/"):
#     for file in tqdm(files):
#         if 'jpg' in file:
#             f.write(os.path.join(root,file)+"\n")
# for root,dirs,files in os.walk("/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/FeiGangJiaGongZhongXin-D2/"):
#     for file in tqdm(files):
#         if 'jpg' in file:
#             f.write(os.path.join(root,file)+"\n")            
# f.close()

# f=open("/home/sma/data960/WYQ/DataTxt/ntzt-ztt/ntzt-ztt_D3.txt",'w')
# for root,dirs,files in os.walk("/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/DongFeiGangKu-D3/"):
#     for file in tqdm(files):
#         if 'jpg' in file:
#             f.write(os.path.join(root,file)+"\n")
# for root,dirs,files in os.walk("/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/FeiGangJiaGongZhongXin-D3/"):
#     for file in tqdm(files):
#         if 'jpg' in file:
#             f.write(os.path.join(root,file)+"\n")            
# f.close()

# f=open("/home/sma/data960/WYQ/DataTxt/ntzt-ztt/ntzt-ztt_D4.txt",'w')
# for root,dirs,files in os.walk("/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/DongFeiGangKu-D4/"):
#     for file in tqdm(files):
#         if 'jpg' in file:
#             f.write(os.path.join(root,file)+"\n")
# for root,dirs,files in os.walk("/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/FeiGangJiaGongZhongXin-D4/"):
#     for file in tqdm(files):
#         if 'jpg' in file:
#             f.write(os.path.join(root,file)+"\n")            
# f.close()

# f=open("/home/sma/data960/WYQ/DataTxt/ntzt-ztt/ntzt-ztt_D5.txt",'w')
# for root,dirs,files in os.walk("/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/DongFeiGangKu-D5/"):
#     for file in tqdm(files):
#         if 'jpg' in file:
#             f.write(os.path.join(root,file)+"\n")
# for root,dirs,files in os.walk("/home/sma/data960/WYQ/WZCData/NanTongZhongTian-ztt/pic/FeiGangJiaGongZhongXin-D5/"):
#     for file in tqdm(files):
#         if 'jpg' in file:
#             f.write(os.path.join(root,file)+"\n")            
# f.close()
