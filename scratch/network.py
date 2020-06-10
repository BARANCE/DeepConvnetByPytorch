# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    """ネットワークモデルの基底クラス
    """
    def __init__( self ):
        """コンストラクタ
        """
        super().__init__()
        self.layers = []
    
    def forward( self, x ):
        """順方向伝播

        Args:
            x (torch.Tensor): 入力データ. ミニバッチ単位で入力すること.

        Returns:
            torch.Tensor: 順方向伝播によって得られるスコア
        """
        for layer in self.layers:
            x = layer.forward( x )
        return x
    
    def save( self, path='./model.pth' ):
        """モデルが持つ重み情報をファイルに記録する

        Args:
            path (str, optional): 保存先ファイル名. Defaults to './model.pth'.
        """
        torch.save( self.state_dict(), path )
    
    def load( self, path='./model.pth' ):
        """ファイルからモデルが持つ重み情報を取得する

        Args:
            path (str, optional): 取得先ファイル名. Defaults to './model.pth'.
        """
        self.load_state_dict( torch.load( path ) )

class DeepConvNet(BaseModel):
    """MNIST画像分類用のDeepLearningネットワークモデル
    """
    def __init__(
        self,
        input_dim = (1, 28, 28), # 入力データの次元(チャンネル数, 高さ, 横幅)
        conv_param_1 = {
            'filter_num': 16, # フィルタのチャンネル数(FN)
            'filter_size': 3, # フィルタサイズ(正方形)(FH, FW)
            'pad': 1, # パディング(入力データの周囲を0で埋める量)
            'stride': 1 # ストライド(フィルタの適用間隔)
        },
        conv_param_2 = { # 2層目のConv
            'filter_num': 16,
            'filter_size': 3,
            'pad': 1,
            'stride': 1
        },
        conv_param_3 = { # 3層目のConv
            'filter_num': 32,
            'filter_size': 3,
            'pad': 1,
            'stride': 1
        },
        conv_param_4 = { # 4層目のConv
            'filter_num': 32,
            'filter_size': 3,
            'pad': 2,
            'stride': 1
        },
        conv_param_5 = { # 5層目のConv
            'filter_num': 64,
            'filter_size': 3,
            'pad': 1,
            'stride': 1
        },
        conv_param_6 = { # 6層目のConv
            'filter_num': 64,
            'filter_size': 3,
            'pad': 1,
            'stride': 1
        },
        hidden_size = 50, # 7層目のaffine
        output_size = 10 # 8層目のaffine(出力)
    ):
        """コンストラクタ

        Args:
            input_dim (tuple, optional): 入力データの次元(ミニバッチの次元は除く). Defaults to (1, 28, 28).
            conv_param_1 (dict, optional): 1層目のConvolutionレイヤパラメータ. Defaults to { 'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1 }.
            conv_param_2 (dict, optional): 2層目のConvolutionレイヤパラメータ. Defaults to { 'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1 }.
            conv_param_3 (dict, optional): 3層目のConvolutionレイヤパラメータ. Defaults to { 'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1 }.
            conv_param_4 (dict, optional): 4層目のConvolutionレイヤパラメータ. Defaults to { 'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1 }.
            conv_param_5 (dict, optional): 5層目のConvolutionレイヤパラメータ. Defaults to { 'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1 }.
            conv_param_6 (dict, optional): 6層目のConvolutionレイヤパラメータ. Defaults to { 'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1 }.
            hidden_size (int, optional): 7層目のAffieレイヤ適用後のノード数. Defaults to 50.
        """
        super().__init__()
        #pylint: disable=no-member
        self.device = torch.device('cuda')

        conv_params = [
            conv_param_1, conv_param_2, conv_param_3,
            conv_param_4, conv_param_5, conv_param_6
        ]

        # Poolingのパラメータ
        pool_window_size = 2
        pool_stride = 2
        # Dropoutのパラメータ
        dropout_ratio = 0.5

        # 3回目のpooling層を抜けた後の1チャンネルあたりのニューロン数
        # このネットワークでは2x2プーリングを用いるため、Pool層を抜けるたびに
        # 入力サイズは半減する。
        # (割り切れない場合は、ceilで切り上げる)
        # なお今回、conv層では入力サイズは変化しない。
        last_pool_size = torch.Tensor([28]).to(self.device)
        for _ in range(3):
            last_pool_size = last_pool_size * 0.5
            #pylint: disable=no-member
            last_pool_size = torch.ceil( last_pool_size )
        last_pool_size = int(last_pool_size.item())
        
        # 1層
        idx = 0
        self.conv1 = nn.Conv2d(
            in_channels=input_dim[0],
            out_channels=conv_params[idx]['filter_num'],
            kernel_size=conv_params[idx]['filter_size'],
            stride=conv_params[idx]['stride'],
            padding=conv_params[idx]['pad']
        )
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        self.relu1 = nn.ReLU()
        idx += 1

        # 2層
        self.conv2 = nn.Conv2d(
            in_channels=conv_params[idx - 1]['filter_num'],
            out_channels=conv_params[idx]['filter_num'],
            kernel_size=conv_params[idx]['filter_size'],
            stride=conv_params[idx]['stride'],
            padding=conv_params[idx]['pad']
        )
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=pool_window_size,
            stride=pool_stride
        )
        idx += 1

        # 3層
        self.conv3 = nn.Conv2d(
            in_channels=conv_params[idx - 1]['filter_num'],
            out_channels=conv_params[idx]['filter_num'],
            kernel_size=conv_params[idx]['filter_size'],
            stride=conv_params[idx]['stride'],
            padding=conv_params[idx]['pad']
        )
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        self.relu3 = nn.ReLU()
        idx += 1

        # 4層
        self.conv4 = nn.Conv2d(
            in_channels=conv_params[idx - 1]['filter_num'],
            out_channels=conv_params[idx]['filter_num'],
            kernel_size=conv_params[idx]['filter_size'],
            stride=conv_params[idx]['stride'],
            padding=conv_params[idx]['pad']
        )
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size=pool_window_size,
            stride=pool_stride
        )
        idx += 1

        # 5層
        self.conv5 = nn.Conv2d(
            in_channels=conv_params[idx - 1]['filter_num'],
            out_channels=conv_params[idx]['filter_num'],
            kernel_size=conv_params[idx]['filter_size'],
            stride=conv_params[idx]['stride'],
            padding=conv_params[idx]['pad']
        )
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        self.relu5 = nn.ReLU()
        idx += 1

        # 6層
        self.conv6 = nn.Conv2d(
            in_channels=conv_params[idx - 1]['filter_num'],
            out_channels=conv_params[idx]['filter_num'],
            kernel_size=conv_params[idx]['filter_size'],
            stride=conv_params[idx]['stride'],
            padding=conv_params[idx]['pad']
        )
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(
            kernel_size=pool_window_size,
            stride=pool_stride
        )
        idx += 1

        # 7層
        self.affine1 = nn.Linear(
            in_features=conv_params[idx - 1]['filter_num'] * last_pool_size * last_pool_size,
            out_features=hidden_size
        )
        nn.init.kaiming_normal_(self.affine1.weight)
        nn.init.constant_(self.affine1.bias, 0)
        self.relu7 = nn.ReLU()
        self.dropout1 = nn.Dropout(
            p=dropout_ratio
        )
        idx += 1

        # 8層
        self.affine2 = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )
        nn.init.kaiming_normal_(self.affine2.weight)
        nn.init.constant_(self.affine2.bias, 0)
        self.dropout2 = nn.Dropout(
            p=dropout_ratio
        )
        idx += 1

        self.layers = [
            self.conv1, self.relu1,
            self.conv2, self.relu2, self.pool1,
            self.conv3, self.relu3,
            self.conv4, self.relu4, self.pool2,
            self.conv5, self.relu5,
            self.conv6, self.relu6, self.pool3,
            self.affine1, self.relu7, self.dropout1,
            self.affine2, self.dropout2
        ]
        
    def forward( self, x ):
        """順方向伝播

        Args:
            x (torch.Tensor): 入力データ. ミニバッチ単位で入力すること.

        Returns:
            torch.Tensor: 順方向伝播によって得られるスコア
        """
        # 1層
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        # 2層
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool1.forward(x)
        # 3層
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        # 4層
        x = self.conv4.forward(x)
        x = self.relu4.forward(x)
        x = self.pool2.forward(x)
        # 5層
        x = self.conv5.forward(x)
        x = self.relu5.forward(x)
        # 6層
        x = self.conv6.forward(x)
        x = self.relu6.forward(x)
        x = self.pool3.forward(x)
        # 2D層から1D層に移るための変換作業
        x_shape = x.size()
        neuron_num = x_shape[1] * x_shape[2] * x_shape[3]
        x = x.view( -1, neuron_num )
        # 7層
        x = self.affine1.forward(x)
        x = self.relu7.forward(x)
        x = self.dropout1(x)
        # 8層
        x = self.affine2.forward(x)
        x = self.dropout2(x)
        
        return x

