import torch
import sys
from config import Config
from torch import optim
import torch.nn as nn
from init_bert import build_dataloader
from tqdm import tqdm
from sklearn.metrics import classification_report
from model2dev import model2dev

# 添加分类层
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=2):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] token 的输出
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)
if __name__ == '__main__':
    config = Config()

    bert_model = config.bert_model.to(config.device)

    model = BertClassifier(bert_model).to(config.device)

    train_loader, test_loader, dev_loader = build_dataloader()

    # 构建优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    loss_f = nn.CrossEntropyLoss()
    best_dev_f1 = 0.0

    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        loss_total = 0.
        train_pres, train_labels = [], []

        print(f'\nEpoch {epoch + 1}/{config.epochs}', flush=True)

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f'训练进度')):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # 前向传播
            y_pre = model(input_ids, attention_mask)

            # 梯度清零
            optimizer.zero_grad()

            # 计算损失
            loss = loss_f(y_pre, labels)

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            loss_total += loss.item()
            pres = torch.argmax(y_pre, dim=1)
            train_pres.extend(pres.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"Train Loss: {loss_total / len(train_loader):.4f}")

        report, f1score, accuracy, precision = model2dev(model, dev_loader, config.device)
        print(f"Dev F1: {f1score:.4f}")
        print(f"Dev Accuracy: {accuracy:.4f}")

        if f1score > best_dev_f1:
            best_dev_f1 = f1score
            torch.save(model.state_dict(), config.train_model_bert)
            print("模型保存！！")
        # 7.11 计算并打印训练集的分类报告
        train_report = classification_report(train_labels, train_pres,
                                             target_names=config.class_list, output_dict=True)
        print(train_report)



