# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class MnistProvider:
    """MNISTデータセットを準備するクラス
    """
    def __init__( self, root='./data' ):
        """コンストラクタ

        Args:
            root (str, optional): データセットのパス. このパスにデータセットが存在しない場合はデータセットをダウンロードする. Defaults to './data'.
        """
        # データセットの値の正規化設定
        mean = (0.5,)
        std = (0.5,)
        normalize = transforms.Normalize( mean, std )
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        # 学習用データセット
        self.trainset = datasets.MNIST(
            root=root,
            train=True,
            transform=transform,
            download=True
        )
        # テスト用データセット
        self.testset = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transform
        )
        self.train_size = self.trainset.data.size()
        self.test_size = self.testset.data.size()

        self.num_workers = 2

    def load_train( self, batch_size=100 ):
        """学習用データセットのloaderを取得する

        Args:
            batch_size (int, optional): ミニバッチサイズ. Defaults to 100.

        Returns:
            torch.utils.data.DataLoader: ミニバッチ単位でデータセットを提供するloaderオブジェクト
        """
        loader = data.DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader
    
    def load_test( self, batch_size=100 ):
        """テスト用データセットのloaderを取得する

        Args:
            batch_size (int, optional): ミニバッチサイズ. Defaults to 100.

        Returns:
            torch.utils.data.DataLoader: ミニバッチ単位でデータセットを提供するloaderオブジェクト
        """
        loader = data.DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader
    
    def load( self, batch_size=100 ):
        """学習用・テスト用データセットのloaderをそれぞれ取得する

        Args:
            batch_size (int, optional): ミニバッチサイズ. Defaults to 100.

        Returns:
            train (torch.utils.data.DataLoader): 学習データのloader
            test (torch.utils.data.DataLoader): テストデータのloader
        """
        train = self.load_train( batch_size=batch_size )
        test = self.load_test( batch_size=batch_size )
        return train, test
