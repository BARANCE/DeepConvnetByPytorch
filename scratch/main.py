# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from network import DeepConvNet
from trainer import Trainer
from provide import MnistProvider
from view import ImgViewer
from param import ParamDeepConvNet

class Main:
    """アプリケーション本体を実現するクラス
    """
    def __init__( self, weight_path=None, params_path=None ):
        """コンストラクタ

        Args:
            weight_path (str, optional): 重み情報が格納されたファイルのパス. Defaults to None.
            params_path (str, optional): ハイパーパラメータ情報が格納されたファイルのパス. Defaults to None.
        """
        # ハイパーパラメータの初期化
        self.params = ParamDeepConvNet()
        # パスが指定されている場合は、そのデータを読み込む
        if params_path is not None:
            self.params.load( path=params_path )
        params_dict = self.params.get()
        learning_rate = params_dict['learning_rate']
        batch_size = params_dict['batch_size']
        max_epoch = params_dict['max_epoch']

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

        # パスが指定されている場合は、学習をせず読み込んだデータを使用する
        if weight_path is not None:
            self.model.load( path=weight_path )
        else:
            # 学習実行
            trainloader = self.provider.load_train( batch_size=batch_size )
            self.trainer.fit( trainloader, max_epoch=max_epoch )
            # 学習結果を保存
            self.model.save() # 重みデータ
            self.params.save() # ハイパーパラメータ

        # 評価実行
        testloader = self.provider.load_test( batch_size=batch_size )
        self.trainer.valid( testloader )

if __name__ == '__main__':
    """エントリポイント
    """
    a = Main()