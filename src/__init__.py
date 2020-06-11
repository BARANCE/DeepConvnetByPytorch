# -*- coding: utf-8 -*-

import sys, os
# srcディレクトリ内にあるモジュールから、同ディレクトリにあるモジュールへのimportを有効にする
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))