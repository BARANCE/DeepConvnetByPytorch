# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import torch

class Trainer:
    """ネットワークモデルを使用して、学習と評価を行う学習器クラス
    """
    def __init__( self, model, optimizer, creterion, device='cpu' ):
        """コンストラクタ

        Args:
            model (BaseModel): ネットワークモデル
            optimizer (torch.optim.Optimizer): 最適化手法
            creterion (torch.nn.Loss): 損失関数
            device (str, optional): 演算を実行するデバイス. Defaults to 'cpu'.
        """
        #pylint: disable=no-member
        self.device = None
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.model = model.to( self.device ) # ネットワークモデル
        self.optimizer = optimizer # 最適化手法
        self.creterion = creterion # 損失関数

        self.loss_list = []
        self.eval_interval = None
    
    def fit(
        self,
        train_loader,
        max_epoch=1
    ):
        """学習実行

        Args:
            train_loader (torch.utils.data.DataLoader): 学習データを提供するloaderオブジェクト
            max_epoch (int, optional): 最大epoch数. この数値だけ学習データ1セットを繰り返し学習させる. Defaults to 1.
        """
        # networkを学習モードに設定
        self.model.train()
        # 1epochあたりの反復回数
        iter_per_epoch = len(train_loader)
        # 学習状態確認用にログ出力するための間隔を決める値
        self.eval_interval = max(int(iter_per_epoch / 10), 1)
        # 学習開始時の時間(ログのタイムスタンプに使用)
        start_time = time.time()
        
        # データセットごとの学習をmax_epoch回繰り返し行う
        for epoch in range( max_epoch ):
            # このepochでの損失の合計値
            running_loss = 0.0
            # 学習用loaderからbatch_size分のデータを取り出す
            for idx_input, input_data in enumerate( train_loader, 0 ):
                images, labels = input_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 順伝播
                outputs = self.model(images)
                # 損失を計算
                loss = self.creterion( outputs, labels )
                
                # 逆伝播
                loss.backward()
                
                # 勾配を更新
                self.optimizer.step()
                
                # このiterationでの損失合計値を追加
                running_loss += loss.item()
                # eval_interval回数反復実行後、学習進捗をログ出力
                if idx_input % self.eval_interval == (self.eval_interval - 1):
                    avg_loss = running_loss / self.eval_interval
                    elapsed_time = time.time() - start_time
                    print(
                        '| epoch: %3d | iter: %5d | time: %5d[s] | loss: %.3f'
                        % (epoch + 1, idx_input + 1, elapsed_time, avg_loss)
                    )
                    self.loss_list.append(float(avg_loss))
                    running_loss = 0.0
        
        print('Finished Training')
    
    def valid(
        self,
        test_loader
    ):
        """評価実行

        Args:
            test_loader (torch.utils.data.DataLoader): テストデータを提供するloaderオブジェクト

        Returns:
            val_loss (float): 損失の平均値
            val_acc (float): 正答率
        """
        # networkをテストモードに設定
        self.model.eval()
        
        running_loss = 0.0 # 損失の合計値
        correct = 0 # 正答した個数
        total = 0 # テストしたデータの個数
        
        # 評価ステップなので勾配は記録しない
        with torch.no_grad():
            # 評価用loaderからbatch_size分のデータを取り出す
            for _, input_data in enumerate( test_loader ):
                images, labels = input_data
                images = images.to( self.device )
                labels = labels.to( self.device )
                
                # 推論
                outputs = self.model( images )
                # 損失を計算
                loss = self.creterion( outputs, labels )
                
                # このiterationでの損失合計値を追加
                running_loss += loss.item()
                
                # スコアが最大となったオフセットを取得
                # maxメソッドは、第一戻り値が最大値、第二戻り値が最大となったオフセット
                #pylint: disable=no-member
                _, max_offset = torch.max( outputs, 1 )
                correct += (max_offset == labels).sum().item() # このiterationの正答数
                total += labels.size(0) # このiterationでテストしたラベルの個数
        
        val_loss = running_loss / len(test_loader) # 損失の平均値
        val_acc = float(correct) / total # 正答率
        
        print('val_loss: %.3f, val_acc: %.3f' % (val_loss, val_acc))
        
        return val_loss, val_acc            
    
    def plot ( self, ylim=None, path=None ):
        """学習結果をmatplotlibでグラフ化し表示またはファイルに保存する

        Args:
            ylim (list, optional): y軸の表示範囲. Defaults to None.
            path (str, optional): グラフファイルの出力先パス. Noneの場合はファイル出力せず画面に表示する. Defaults to None.
        """
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot( x, self.loss_list, label='train' )
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()