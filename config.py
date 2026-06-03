from transformers import BertModel, BertTokenizer
import torch

class Config(object):
    def __init__(self):
        self.data_path = './data'
        self.train_path = './data/train.csv'
        self.test_path = './data/test.csv'
        self.dev_path = './data/dev.csv'
        self.class_path = './data/class.txt'

        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("✓ 使用 Apple Silicon MPS 加速")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("✓ 使用 CUDA GPU")
        else:
            self.device = 'cpu'
            print("⚠️ 使用 CPU (较慢)")

        if self.device == 'mps':
            self.batch_size = 16
        elif self.device == 'cuda':
            self.batch_size = 32
        else:
            self.batch_size = 4

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.embed_size = 256
        self.hidden_size_lstm = 512
        self.num_layers = 4
        self.dropout = 0.3
        self.save_model_path = "./models_save"
        self.train_model_bert = self.save_model_path + '/bert_train.pt'

        self.class_list = ['负向评论', '正向评论']

        # 降低学习率,增加epoch
        self.lr = 1e-5  # 从 2e-5 降到 1e-5
        self.epochs = 6  # 从 4 增加到 6
