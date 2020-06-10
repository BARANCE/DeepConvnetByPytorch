# -*- coding: utf-8 -*-

import pickle

class Parameters:
    """ハイパーパラメータを管理する基底クラス
    """
    def __init__( self, params ):
        """コンストラクタ

        Args:
            params (dict): ハイパーパラメータ一覧のkey-valueオブジェクト
        """
        self.params = params
        self.default_path = './params.pkl'
    
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
        
        with open( path, 'wb' ) as fd:
            pickle.dump( self.params, fd )
    
    def load( self, path=None ):
        """ファイルからハイパーパラメータを取得する

        Args:
            path (str, optional): 取得先ファイルのパス. Defaults to None.
        """
        if path is None:
            path = self.default_path
        
        with open( path, 'rb' ) as fd:
            self.params = pickle.load(fd)


class ParamDeepConvNet(Parameters):
    """DeepConvNetの学習で使用するハイパーパラメータを管理するクラス
    """
    def __init__( self ):
        """コンストラクタ
        """
        params = {
            'learning_rate': 0.001,
            'batch_size': 100,
            'max_epoch': 5
        }
        super().__init__( params )
        self.default_path = './params_deepconvnet.pkl'