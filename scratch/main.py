# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from network import DeepConvNet
from trainer import Trainer
from provide import MnistProvider
from view import ImgViewer

class Main:
    """アプリケーション本体を実現するクラス
    """
    def __init__( self, train_flg=False ):
        """コンストラクタ

        Args:
            train_flg (bool, optional): 学習実行時はTrue, 評価実行時はFalse. Defaults to False.
        """
        learning_rate = 0.001
        batch_size = 100
        max_epoch = 5
        
        # 各種インスタンスを構築する
        self.provider = MnistProvider() # データセット
        self.model = DeepConvNet() # ネットワークモデル
        self.criterion = nn.CrossEntropyLoss() # 損失関数
        self.optimizer = optim.Adam( # 最適化手法
            self.model.parameters(),
            lr=learning_rate
        )
        self.trainer = Trainer( # 学習器
            self.model,
            self.optimizer,
            self.criterion
        )
        
        if train_flg:
            # 学習実行
            trainloader = self.provider.load_train( batch_size=batch_size )
            self.trainer.fit( trainloader, max_epoch=max_epoch )
            self.model.save()
        else:
            # 評価実行
            self.model.load()
            testloader = self.provider.load_test( batch_size=batch_size )
            self.trainer.valid( testloader )

if __name__ == '__main__':
    """エントリポイント
    """
    a = Main()