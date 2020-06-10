# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torchvision

class ImgViewer:
    """画像データをmatplotlibを使用して画面に表示する
    """
    def __init__( self, dataloader ):
        """コンストラクタ

        Args:
            dataloader (torch.utils.data.DataLoader): 画像データのloader
        """
        self.iter = iter(dataloader)
    
    def imshow( self, img ):
        """imgに指定された1個の画像データを表示する

        Args:
            img (torch.Tensor): 画像データを表すTensor
        """
        img = img / 2 + 0.5 # データ範囲の調整(0-1に収める)
        npimg = img.numpy()
        npimg = np.transpose( npimg, (1, 2, 0) )
        plt.imshow( npimg )
        plt.axis('off')
        plt.show()
    
    def show( self ):
        """loaderからデータを1セット分ロードし、それらを画面に表示する
        """
        # 次のデータセットを読み出す
        images, _ = self.iter.next()
        # データをグリッド上に配置する加工
        joinimg = torchvision.utils.make_grid( images, nrow=5, padding=1 )
        # 加工した画像を表示
        self.imshow( joinimg )