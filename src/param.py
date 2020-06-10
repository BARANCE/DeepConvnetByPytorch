# -*- coding: utf-8 -*-

import json

class Parameters:
    """ハイパーパラメータを管理する基底クラス
    """
    def __init__( self, params ):
        """コンストラクタ

        Args:
            params (dict): ハイパーパラメータ一覧のkey-valueオブジェクト
        """
        self.params = params
        self.default_path = './params.json'
    
    def get( self ):
        """保存されているハイパーパラメータを取得する

        Returns:
            dict: ハイパーパラメータ
        """
        return self.params
    
    def save( self, path=None ):
        """ハイパーパラメータをファイルに保存する

        Args:
            path (str, optional): 保存先ファイルのパス. Defaults to None.
        """
        if path is None:
            path = self.default_path
        
        with open( path, 'w' ) as fd:
            json.dump( self.params, fd, indent=4 )
    
    def load( self, path=None ):
        """ファイルからハイパーパラメータを取得する

        Args:
            path (str, optional): 取得先ファイルのパス. Defaults to None.
        """
        if path is None:
            path = self.default_path
        
        with open( path, 'r' ) as fd:
            self.params = json.load(fd)


class ParamDeepConvNet(Parameters):
    """DeepConvNetの学習で使用するハイパーパラメータを管理するクラス
    """
    def __init__( self ):
        """コンストラクタ
        """
        params = {
            'learning_rate': 0.001,
            'batch_size': 100,
            'max_epoch': 1,
            'weight_path': './model_deepconvnet.pth',
            'use_weight': False,
            'plot_path': './plot_deepconvnet.png',
            'device': 'cpu'
        }
        super().__init__( params )
        self.default_path = './params_deepconvnet.json'