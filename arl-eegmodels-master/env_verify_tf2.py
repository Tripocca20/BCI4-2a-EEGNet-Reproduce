# -*- coding: gbk -*-
from EEGModels import EEGNet  # 作者原版EEGNet（TF2.x版本）
import sys,os
import tensorflow as tf
import numpy as np
import mne
import sklearn
import matplotlib

# -------------------------- 1. 环境版本验证 --------------------------
print("===== 环境版本验证 =====")
print(f"Python版本: {tf.sys.version.split()[0]}")
print(f"TensorFlow版本: {tf.__version__}")
print(f"MNE版本: {mne.__version__}")
print(f"scikit-learn版本: {sklearn.__version__}")
print(f"matplotlib版本: {matplotlib.__version__}")

# 验证核心版本要求
assert "3.7" in tf.sys.version or "3.8" in tf.sys.version, "Python版本需为3.7/3.8"
assert "2." in tf.__version__, "TensorFlow版本需为2.x"
assert mne.__version__ >= "0.17.1", "MNE版本需≥0.17.1"
print("? 所有环境版本验证通过！\n")

# -------------------------- 2. 生成模拟EEG数据 --------------------------
print("===== 生成模拟EEG数据 =====")
# 匹配BCI 2a数据集维度：[样本数, 通道数, 时间点, 1]
X_train = np.random.randn(16, 22, 1000, 1).astype(np.float32)
y_train = np.random.randint(0, 4, size=(16,))
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=4)
print(f"训练数据维度: {X_train.shape}")
print(f"标签维度: {y_train_one_hot.shape}")
print("? 模拟数据生成完成！\n")

# -------------------------- 3. 构建EEGNet模型 --------------------------
print("===== 构建EEGNet模型 =====")
model = EEGNet(
    nb_classes=4,
    Chans=22,
    Samples=1000,
    dropoutRate=0.5,
    kernLength=64,
    F1=8,
    D=2,
    F2=16,
    dropoutType='Dropout'
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f"模型总参数量: {model.count_params():,} 个")
print("? EEGNet模型构建+编译完成！\n")

# -------------------------- 4. 单次训练+推理验证 --------------------------
print("===== 训练+推理验证 =====")
history = model.fit(
    X_train, y_train_one_hot,
    batch_size=8,
    epochs=1,
    verbose=1
)

test_pred = model.predict(X_train[:2])
print(f"\n推理结果维度: {test_pred.shape}")
print(f"第一个样本预测概率: {np.round(test_pred[0], 4)}")
print(f"第一个样本预测类别: {np.argmax(test_pred[0])}")
print("? 训练+推理验证通过！\n")

# -------------------------- 5. 最终结论 --------------------------
print("? ? ? 所有验证步骤全部完成！")
print("? 你的TF2.x环境完全适配作者新版EEGNet代码！")
print("? 接下来可以直接运行作者的EEG/MEG ERP分类示例脚本了！")