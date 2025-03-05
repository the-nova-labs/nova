# -*- coding: utf-8 -*-
import os

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    device = os.get_environ("DEVICE_OVERRIDE")
    DEVICE = ["cpu" if device=="cpu" else "cuda:0"][0]
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'PDBv2020_PSICHIC')
    BATCH_SIZE = 128
    
