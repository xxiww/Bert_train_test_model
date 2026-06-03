import pandas as pd
from config import Config

config = Config()

with open(config.save_model_path + '/a.txt','w',encoding='utf-8') as f:
    f.write('你好')


sl = {'正向评论': {'precision': 0.8917126779911764, 
                   'recall': 0.8030357100373516,
                   'f1-score': 0.8450542022381894,
                   'support': 42033.0},
      '负向评论': {'precision': 0.8205600589535741,
                   'recall': 0.9023070689737357,
                   'f1-score': 0.8594941881583726,
                   'support': 41958.0},
      'accuracy': 0.8526270671857699,
      'macro avg': {'precision': 0.8561363684723753,
                    'recall': 0.8526713895055437,
                    'f1-score': 0.8522741951982811,
                    'support': 83991.0},
      'weighted avg': {'precision': 0.856168136438156,
                       'recall': 0.8526270671857699,
                       'f1-score': 0.8522677480852332,
                       'support': 83991.0}}