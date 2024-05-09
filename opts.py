import argparse
from email import parser
from pickle import TRUE
def str2bool(str):
	return True if str.lower() == 'true' else False
class args:
    def __init__(self):
        parser=argparse.ArgumentParser(description='输入参数')
        # parser.add_argument('--data_path','-d',help='图片数据集绝对路径',required=TRUE)
        # parser.add_argument('--model_path','-m',help='模型的绝对路径',required=TRUE)
        # parser.add_argument('--prelabel_path','-p',help='完成预标注保存生成的json文件的绝对路径',default=TRUE)
        parser.add_argument('--data_set_txt','-ds',type=str,required=True,help='数据集txt文件路径,包含图片的完整路径名，为jpg格式')
        parser.add_argument('--save_dir','-sd',type=str,required=True,help='训练结果输出目录')     
        parser.add_argument('--train_ratio','-tr',type=float,required=False,default=0.7,help='训练集占比')        
        parser.add_argument('--Cuda','-Cd', type=str2bool, required=False,default=True, help="是否使用Cuda")
        parser.add_argument('--distributed', '-dis', type=str2bool, required=False,default=False, help="用于指定是否使用单机多卡分布式运行,终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡,Windows系统下默认使用DP模式调用所有显卡，不支持DDP。")
        parser.add_argument('--sync_bn','-sb', type=str2bool, required=False,default=False, help="是否使用sync_bn，DDP模式多卡可用")
        parser.add_argument('--fp16','-fp', type=str2bool, required=False,default=False, help="是否使用混合精度训练,可减少约一半的显存、需要pytorch1.7.1以上")
        parser.add_argument('--backbone','-bb', type=str, required=False,default="mobilenet", help="所有使用的主干网络，有mobilenet,xception")
        parser.add_argument('--pretrained','-pre', type=str2bool, required=False,default=False, help="是否使用主干网络的预训练权重")
        parser.add_argument('--model_path','-mp', type=str, required=False,default="model_data/deeplab_mobilenetv2.pth", help="是否使用主干网络的预训练权重")
        parser.add_argument('--downsample_factor','-df', type=int, required=False,default=16, help="下采样的倍数8、16,8下采样的倍数较小、理论上效果更好")
        parser.add_argument('--input_shape','-is', type=int,nargs='+', required=False,default=[360,640], help="模型输入图片大小：[高，宽]")
        parser.add_argument('--Init_Epoch','-IE', type=int, required=False,default=0, help="模型当前开始的训练世代")
        parser.add_argument('--Freeze_Epoch','-FE', type=int, required=False,default=50, help="模型开始冻结训练的训练世代")
        parser.add_argument('--Freeze_batch_size','-Fb', type=int, required=False,default=4, help="模型冻结训练的batch_size")
        parser.add_argument('--UnFreeze_Epoch','-UFE', type=int, required=False,default=100, help="模型总共训练的epoch")
        parser.add_argument('--Unfreeze_batch_size','-Ub', type=int, required=False,default=4, help="模型在解冻后的batch_size")
        parser.add_argument('--Freeze_Train','-FT', type=str2bool, required=False,default=True, help="是否进行冻结训练")
        
        parser.add_argument('--Init_lr','-Il',type=float,required=False,default=7e-3,help="模型的最大学习率")
        parser.add_argument('--optimizer_type','-ot',type=str,required=False,default="sgd",help="使用到的优化器种类，可选的有adam、sgd")
        parser.add_argument('--momentum','-mm',type=float,required=False,default=0.9,help="优化器内部使用到的momentum参数")
        parser.add_argument('--weight_decay','-wd',type=float,required=False,default=1e-4,help="权值衰减，可防止过拟合")
        parser.add_argument('--lr_decay_type','-ld',type=str,required=False,default='cos',help="使用到的学习率下降方式，可选的有step,cos")

        parser.add_argument('--save_period','-sp', type=int, required=False,default=5, help="多少个epoch保存一次权值")

        parser.add_argument('--eval_flag','-ef', type=str2bool, required=False,default=False, help="是否在训练时进行评估，评估对象为验证集")
        parser.add_argument('--eval_period','-ep', type=int, required=False,default=5, help="代表多少个epoch评估一次，不建议频繁的评估")
        
        parser.add_argument('--dice_loss','-dl', type=str2bool, required=False,default=True, help="种类少（几类）时，设置为True")
        parser.add_argument('--focal_loss','-fl', type=str2bool, required=False,default=True, help="是否使用focal loss来防止正负样本不平衡")
    
        parser.add_argument('--num_workers','-nw', type=int, required=False,default=4, help="是否使用多线程读取数据，1代表关闭多线程")
        parser.add_argument('--num_classes','-nc', type=int, required=False,default=2, help="自己需要的分类个数")
        parser.add_argument('--consider_type','-ct',nargs='+',required=False,default=['Sm'],help="都会加有背景类_background，此处只加入需要考虑的分割类，默认有Sm")
        self.parameters= parser.parse_args()