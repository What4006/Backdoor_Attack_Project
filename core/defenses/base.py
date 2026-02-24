import os
import random
import torch
import numpy as np

class Base(object):
    """
    改进后的基类，支持接收 model, criterion 等参数，兼容 ABL_YOLO 的调用。
    """
    def __init__(self, model=None, criterion=None, trainset=None, testset=None, args=None, **kwargs):
        # 1. 保存核心组件
        self.model = model
        self.criterion = criterion
        self.trainset = trainset
        self.testset = testset
        self.args = args
        
        # 2. 设置随机种子 (优先从 args 中读取，没有则默认 0)
        seed = getattr(args, 'seed', 0) if args else 0
        deterministic = getattr(args, 'deterministic', False) if args else False
        self._set_seed(seed, deterministic)

    def _set_seed(self, seed, deterministic):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True