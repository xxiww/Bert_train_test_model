import torch
import time
from config import Config
from train import BertClassifier

config = Config()
device = config.device

# 加载模型
print(f'从 {config.train_model_bert} 加载模型...', flush=True)
model = BertClassifier(config.bert_model.to(device))

# 正确加载 checkpoint - 关键修复
try:
    checkpoint = torch.load(config.train_model_bert, map_location=device)
    # 如果是字典格式(checkpoint),提取 model_state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print('✓ 以 checkpoint 格式加载成功', flush=True)
    else:
        # 如果直接是 state_dict
        model.load_state_dict(checkpoint)
        print('✓ 以 state_dict 格式加载成功', flush=True)
except Exception as e:
    print(f'✗ 加载失败: {e}', flush=True)
    raise

model.to(device).eval()

print('✓ 模型加载成功', flush=True)
print(f'设备: {device}', flush=True)
print(f'类别列表: {config.class_list}', flush=True)

tokenizer = config.tokenizer

def predict_single(text):
    """预测单条文本"""
    tokenize = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    input_ids = tokenize['input_ids'].to(device)
    attention_mask = tokenize['attention_mask'].to(device)
    
    with torch.no_grad():
        start = time.time()
        pre_logits = model(input_ids, attention_mask)
        pre_prob = torch.softmax(pre_logits, dim=1)
        
        pre_ids = torch.argmax(pre_prob, dim=1)
        category = config.class_list[pre_ids.item()]
        confidence = pre_prob[0, pre_ids.item()].item()
        end = (time.time() - start) * 1000
        
        return category, confidence, end

# 测试示例
print('\n' + '='*50)
print('开始测试')
print('='*50)

test_cases = [
    ('这家面真的是绝绝子啊，一定还会来', '预期: 正向'),
    ('太难吃了，再也不来了', '预期: 负向'),
    ('味道一般般吧', '预期: 中性或负向'),
    ('超级好吃！强烈推荐！', '预期: 正向'),
    ('服务态度很差，菜品也不新鲜', '预期: 负向'),
]

for text, expected in test_cases:
    print(f'\n文本: {text}')
    print(f'{expected}', flush=True)
    category, confidence, time_cost = predict_single(text)
    print(f'预测: {category}')
    print(f'置信度: {confidence:.4f}')
    print(f'耗时: {time_cost:.2f}ms', flush=True)

