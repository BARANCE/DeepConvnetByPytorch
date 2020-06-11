# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from network import DeepConvNet
from trainer import Trainer
from provide import MnistProvider
from view import ImgViewer
from param import ParamDeepConvNet

class Application:
    """アプリケーション本体を実現するクラス
    """
    def __init__( self, params_path='./params_deepconvnet.json' ):
        """コンストラクタ

        Args:
            params_path (str, optional): ハイパーパラメータ情報が格納されたファイルのパス. Defaults to None.
        """
        # 動作に必要なインスタンス類
        self.params = None # 各種パラメータが格納されているオブジェクト
        self.provider = None # データセットを提供するオブジェクト
        self.model = None # ネットワークモデル
        self.criterion = None # 損失関数
        self.optimizer = None # 最適化手法
        self.trainer = None # 学習器

        # ハイパーパラメータ
        self.batch_size = None # ミニバッチサイズ
        self.max_epoch = None # 最大epoch数

        # 各種設定情報
        self.params_path = params_path # パラメータファイルのパス
        self.weight_path = None # ネットワークの重みファイルのパス
        self.plot_path = None # 学習結果グラフの出力先パス
        self.b_use_weight = None # 重みファイルを使用するかどうか

        # 準備
        self.prepare()
        # 実行
        self.train() # 学習
        self.eval() # 評価

    def prepare ( self ):
        """学習・評価のための各種データの準備を行う
        """
        # ハイパーパラメータの初期化
        self.params = ParamDeepConvNet()
        # パスが指定されている場合は、そのデータを読み込む
        if self.params_path is not None:
            self.params.load( path=self.params_path )
        params_dict = self.params.get()
        learning_rate = params_dict['learning_rate']
        self.batch_size = params_dict['batch_size']
        self.max_epoch = params_dict['max_epoch']

        # 各種設定情報(オプション)
        # 重みデータのパス
        if 'weight_path' in params_dict:
            self.weight_path = params_dict['weight_path']
        # 重みデータファイルを使用するかどうか
        if 'use_weight' in params_dict:
            self.b_use_weight = params_dict['use_weight']
        else:
            self.b_use_weight = False
        # 結果グラフの保存先
        if 'plot_path' in params_dict:
            self.plot_path = params_dict['plot_path']
        # 演算を実行するデバイス
        device = None
        if 'device' in params_dict:
            device = params_dict['device']

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
            self.criterion,
            device=device
        )

    def train ( self ):
        """学習の実行
        """
        # パスが指定されている場合は、学習をせず読み込んだデータを使用する
        if (self.weight_path is not None) and self.b_use_weight:
            self.model.load( path=self.weight_path )
        else:
            # 学習実行
            trainloader = self.provider.load_train( batch_size=self.batch_size )
            self.trainer.fit( trainloader, max_epoch=self.max_epoch )
            # 結果を表示
            if self.plot_path is not None:
                self.trainer.plot( path=self.plot_path )
            else:
                self.trainer.plot()
            # 学習結果を保存
            if self.weight_path is not None:
                self.model.save( path=self.weight_path ) # 重みデータ
            else:
                self.model.save()
            if self.params_path is not None:
                self.params.save( path=self.params_path ) # ハイパーパラメータ
            else:
                self.params.save()

    def eval ( self ):
        # 評価実行
        testloader = self.provider.load_test( batch_size=self.batch_size )
        self.trainer.valid( testloader )
