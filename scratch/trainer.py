# -*- coding: utf-8 -*-

import torch

class Trainer:
    """ネットワークモデルを使用して、学習と評価を行う学習器クラス
    """
    def __init__( self, model, optimizer, creterion ):
        """コンストラクタ

        Args:
            model (BaseModel): ネットワークモデル
            optimizer (torch.optim.Optimizer): 最適化手法
            creterion (torch.nn.Loss): 損失関数
        """
        #pylint: disable=no-member
        self.device = torch.device('cuda')
        
        self.model = model.to( self.device ) # ネットワークモデル
        self.optimizer = optimizer # 最適化手法
        self.creterion = creterion # 損失関数
    
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
        eval_interval = max(int(iter_per_epoch / 10), 1)
        
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
                if idx_input % eval_interval == (eval_interval - 1):
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, idx_input + 1, running_loss / eval_interval))
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
        