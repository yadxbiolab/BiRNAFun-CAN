# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import StandardScaler
# import os
# import random
# import time
# # 设置设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(24)

# class Projection_Attention(nn.Module):
#     def __init__(self, input_dim, hidden_size, attention_size, num_heads):
#         super(Projection_Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attention_size = attention_size
#         self.num_heads = num_heads

#         self.query_proj = nn.Linear(input_dim, hidden_size * num_heads)
#         self.key_proj = nn.Linear(input_dim, hidden_size * num_heads)
#         self.value_proj = nn.Linear(input_dim, attention_size * num_heads)

#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()

#         query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)
#         key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)
#         value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.attention_size)

#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.hidden_size)
#         attention_probs = self.softmax(attention_scores)

#         context = torch.matmul(attention_probs, value)
#         context = context.view(batch_size, seq_len, self.num_heads * self.attention_size)

#         return context


# class Model(nn.Module):
#     def __init__(self, input_dim):
#         super(Model, self).__init__()
#         self.conv_net1 = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(16, 32, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2))
        
#         self.conv_net2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(64, 128, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.conv_net3 = nn.Sequential(
#             nn.Conv1d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(256, 256, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2))
        
#         self.attention = Projection_Attention(input_dim=256, hidden_size=64, attention_size=32, num_heads=4)
        
#         self.fc1 = nn.Linear(input_dim // 16 * 256, 64)  # 自动适配输入维度
#         self.bn1 = nn.BatchNorm1d(64)
#         self.dropout1 = nn.Dropout(0.5)
        
#         self.fc2 = nn.Linear(64, 32)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.dropout2 = nn.Dropout(0.5)
        
#         self.output = nn.Linear(32, 1)

#     def forward(self, x):
#         x = self.conv_net1(x)
#         x = self.conv_net2(x)
#         x = self.conv_net3(x)
        
#         x = x.permute(0, 2, 1)
#         x = self.attention(x)
        
#         x = x.view(x.size(0), -1)  
        
#         x = self.bn1(nn.ReLU()(self.fc1(x)))
#         x = self.dropout1(x)
        
#         x = self.bn2(nn.ReLU()(self.fc2(x)))
#         x = self.dropout2(x)
        
#         x = torch.sigmoid(self.output(x))
#         return x

# # 加载测试数据
# test_df = pd.read_excel('test(49+kmer)-sample.xlsx')  # 替换为你的测试数据文件路径

# # 确保选择前n-1列为特征，最后一列为标签
# X_test = test_df.iloc[:, :-1].values

# # 加载标准化器并进行数据预处理
# scaler = StandardScaler()
# scaler.mean_ = np.load("scaler_mean.npy", allow_pickle=False)
# scaler.scale_ = np.load("scaler_scale.npy", allow_pickle=False)
# X_test = scaler.transform(X_test)

# # 将数据转换为PyTorch张量，并调整形状以适应卷积层的输入要求
# X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # (num_samples, num_features) -> (num_samples, 1, num_features)

# # 加载训练好的模型
# input_dim = X_test.shape[2]
# model = Model(input_dim)

# # 如果是在GPU上训练的模型，确保加载到GPU上
# if torch.cuda.is_available():
#     model.load_state_dict(torch.load('model2.pth', weights_only=True))
#     model.to(device)
# else:
#     model.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))

# model.eval()

# # 将测试数据移动到相同的设备上
# X_test = X_test.to(device)

# start_time = time.time()

# with torch.no_grad():
#     test_outputs = model(X_test).squeeze()

# end_time = time.time()
# elapsed_time = end_time - start_time

# print(f"预测用时: {elapsed_time:.4f} 秒")

# # 保存预测结果
# predicted_scores = test_outputs.cpu().numpy()
# pd.DataFrame(predicted_scores, columns=['Predicted Scores']).to_csv('5555.csv', index=False)







import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import random
import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve, accuracy_score, f1_score, average_precision_score

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(28)

class Projection_Attention(nn.Module):
    def __init__(self, input_dim, hidden_size, attention_size, num_heads):
        super(Projection_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.num_heads = num_heads
        
        self.query_proj = nn.Linear(input_dim, hidden_size * num_heads)
        self.key_proj = nn.Linear(input_dim, hidden_size * num_heads)
        self.value_proj = nn.Linear(input_dim, attention_size * num_heads)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)
        key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)
        value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.attention_size)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.hidden_size)
        attention_probs = self.softmax(attention_scores)
        
        context = torch.matmul(attention_probs, value)
        context = context.view(batch_size, seq_len, self.num_heads * self.attention_size)
        
        return context


class Model(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(Model, self).__init__()
        self.seq_len = seq_len

        # CNN分支
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Attention分支
        self.attention = Projection_Attention(input_dim=1, hidden_size=64, attention_size=32, num_heads=4)

        # 计算 CNN 输出后的序列长度（原始长度 / 2^3）
        cnn_output_len = seq_len // 8  # 因为有3个MaxPool1d(kernel_size=2)
        cnn_out_features = cnn_output_len * 256

        # Attention 输出：seq_len × (num_heads * attention_size)
        attn_out_features = seq_len * (4 * 32)

        # 合并后的全连接层
        self.fc1 = nn.Linear(cnn_out_features + attn_out_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.5)

        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # x: [B, 1, seq_len]
        cnn_input = x  # shape [B, 1, seq_len]
        attn_input = x.squeeze(1)  # shape [B, seq_len]

        # CNN路径
        cnn_out = self.conv_net(cnn_input)  # [B, 256, seq_len/8]
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # Flatten

        # Attention路径
        attn_input = x.permute(0, 2, 1)  # [B, 1, 133] -> [B, 133, 1]
        attn_out = self.attention(attn_input)  # 输入 shape: [B, 133, 1]
        attn_out = attn_out.view(attn_out.size(0), -1)  # Flatten

        # 拼接两个分支的输出
        x = torch.cat([cnn_out, attn_out], dim=1)  # [B, total_features]

        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout1(x)

        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.sigmoid(self.output(x))
        return x
    


# 加载测试数据
test_df = pd.read_excel('test(49+kmer)-sample.xlsx')  # 替换为你的测试数据文件路径

# 确保选择前n-1列为特征，最后一列为标签
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1]
# 加载标准化器并进行数据预处理
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy", allow_pickle=False)
scaler.scale_ = np.load("scaler_scale.npy", allow_pickle=False)
X_test = scaler.transform(X_test)

# 将数据转换为PyTorch张量，并调整形状以适应卷积层的输入要求
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # (num_samples, num_features) -> (num_samples, 1, num_features)
y_test = torch.tensor(y_test, dtype=torch.float32)
# 加载训练好的模型
input_dim = X_test.shape[2]
model = Model(input_dim=X_test.shape[2], seq_len=X_test.shape[2])

# 如果是在GPU上训练的模型，确保加载到GPU上
if torch.cuda.is_available():
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.to(device)
else:
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

model.eval()

# # 将测试数据移动到相同的设备上
# X_test = X_test.to(device)

# start_time = time.time()

# with torch.no_grad():
#     test_outputs = model(X_test).squeeze()

# end_time = time.time()
# elapsed_time = end_time - start_time

# print(f"预测用时: {elapsed_time:.4f} 秒")

# # 保存预测结果
# predicted_scores = test_outputs.cpu().numpy()
# pd.DataFrame(predicted_scores, columns=['Predicted Scores']).to_csv('5555.csv', index=False)



# 将测试数据移动到相同的设备上
# X_test = X_test.to(device)
# y_test = y_test.to(device)


# # 评估模型在测试集上的表现
# with torch.no_grad():
#     test_outputs = model(X_test).squeeze()
#     test_loss = torch.nn.BCELoss()(test_outputs, y_test).item()
#     test_predictions = test_outputs.round().cpu().numpy()

# # 计算AUC
# auc_score = roc_auc_score(y_test.cpu().numpy(), test_outputs.cpu().numpy())

# # 计算MCC
# mcc = matthews_corrcoef(y_test.cpu().numpy(), test_predictions)

# # 计算敏感性和特异性
# tn, fp, fn, tp = confusion_matrix(y_test.cpu().numpy(), test_predictions).ravel()
# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)

# # 计算准确率 (ACC)
# acc = accuracy_score(y_test.cpu().numpy(), test_predictions)

# # 计算F1分数
# f1 = f1_score(y_test.cpu().numpy(), test_predictions)
# aupr = average_precision_score(y_test.cpu().numpy(), test_outputs.cpu().numpy())
# # 打印和保存结果
# print(f'Test Loss: {test_loss:.4f}')
# print(f'Test AUC: {auc_score:.4f}')
# print(f'Test MCC: {mcc:.4f}')
# print(f'Sensitivity: {sensitivity:.4f}')
# print(f'Specificity: {specificity:.4f}')
# print(f'Test ACC: {acc:.4f}')
# print(f'Test F1: {f1:.4f}')
# print(f'Test AUPR: {aupr:.4f}')


# # 创建结果数据框
# results = pd.DataFrame({
#     'Test Loss': [test_loss],
#     'Test AUC': [auc_score],
#     'Test MCC': [mcc],
#     'Sensitivity': [sensitivity],
#     'Specificity': [specificity],
#     'Test ACC': [acc],
#     'Test F1': [f1],
#     'Test AUPR': aupr
# })

# # 保存预测结果和指标到Excel文件
# with pd.ExcelWriter('test_results_5.xlsx') as writer:
#     results.to_excel(writer, sheet_name='Metrics', index=False)
#     predictions_df = pd.DataFrame(test_predictions, columns=['Predicted Probability'])
#     predictions_df.to_excel(writer, sheet_name='Predictions', index=False)

# print("Test results written to bingxing.xlsx")



X_test = X_test.to(device)
y_test = y_test.to(device)

# 获取模型预测结果
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_predictions = test_outputs.cpu().numpy()  # 获取预测概率分数

# 创建预测结果DataFrame
predictions_df = pd.DataFrame({
    'True_Label': y_test.cpu().numpy(),
    'Predicted_Score': test_predictions
})

# 保存预测结果到Excel文件
predictions_df.to_excel('prediction_scores.xlsx', index=False)

print("Prediction scores saved to prediction_scores.xlsx")