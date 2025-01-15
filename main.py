
from model.skynet import SkyNet
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    skynet = SkyNet()
    skynet.inference('特朗普是谁？', False)
    skynet.inference('求解力的大小是2，力的作用时间是3s，计算冲量', False)
