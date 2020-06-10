# -*- coding: utf-8 -*-

import sys
from src.application import Application

if __name__ == '__main__':
    """エントリポイント
    """
    args = sys.argv
    if len(args) >= 2:
        Application( params_path=args[1] )
    else:
        Application()