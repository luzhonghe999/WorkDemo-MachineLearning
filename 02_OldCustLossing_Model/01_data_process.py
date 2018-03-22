
# -*- coding: utf-8 -*-
import sys, os
pwd = sys.path[0]    # 获取当前执行脚本的位置
print(pwd)
print(os.path.abspath(os.path.join(pwd, os.pardir, os.pardir)))