# -*- coding: gbk -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import numpy as np
from scipy.io import loadmat
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import time



# ===================== 步骤1：导入依赖 =====================
EEG_MODELS_PATH = "F:\\Document\\python\\BCI\\arl-eegmodels-master"
sys.path.append(EEG_MODELS_PATH)
from EEGModels import EEGNet
import mne
from mne import io

# ===================== 步骤2：配置文件路径 =====================
GDF_FILE_PATH = r"F:\\Document\\python\\BCI\\BCI_Competition_IV_2a\\dataset\\A01T.gdf"  # gdf数据文件
MAT_LABEL_PATH = r"F:\\Document\\python\\BCI\\BCI_Competition_IV_2a\\label\\A01T.mat"  # 极简版mat标签文件
sfreq_target = 128  # 目标重采样率
sfreq_original = 250  # BCI 2a官方原始采样率（固定不变）
trial_interval = 4.5  # 官方试次间隔（固定4.5s，含基线+运动想象）
n_trials = 288  # 官方固定试次数（288个，4类各72个）

# ===================== 步骤3：加载.gdf+基础预处理（MNE核心操作） =====================
# 3.1 加载gdf，删除EOG通道，保留22个EEG通道
raw = io.read_raw_gdf(GDF_FILE_PATH, preload=True, verbose=False)
raw = raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
print("筛选后EEG通道数：", len(raw.ch_names))  # 验证：22

# 3.2 8-30Hz带通滤波（保留运动想象核心频段，去除伪迹）+ 重采样至128Hz
raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)
raw.resample(sfreq=sfreq_target, verbose=False)
sfreq = raw.info['sfreq']
print("重采样后最终采样率：", sfreq, "Hz")  # 验证：128.0

# 3.3 获取gdf文件重采样后的总采样点数（关键：用于事件时间对齐）
total_samples_128 = raw.n_times
print("gdf文件总采样点数（128Hz）：", total_samples_128)

# ===================== 步骤4：加载极简版.mat，提取真实类别标签 =====================
mat_data = loadmat(MAT_LABEL_PATH, squeeze_me=True, struct_as_record=False)
classlabel = mat_data['classlabel']  # 提取唯一的标签键名
# 标签格式适配：确保为一维数组，映射为官方事件码（1→769,2→770,3→771,4→772）
event_labels = classlabel.flatten()  # 展平为一维，避免维度问题
event_code_map = {1:769, 2:770, 3:771, 4:772}
event_codes = np.array([event_code_map[lab] for lab in event_labels])
print("提取真实标签数：", len(event_codes))  # 验证：288
print("标签分布（4类各72个）：", np.bincount(event_labels))  # 验证：[0 72 72 72 72]

# ===================== 步骤5：精准生成与gdf对齐的事件触发点（核心！替代缺失标注） =====================
# 5.1 计算原始250Hz下的总试次时长（288个试次×4.5s/试次）
total_trial_duration = n_trials * trial_interval
# 5.2 计算250Hz下的有效数据起始点（官方实验：前2s为基线，从2s开始触发第一个试次）
onset_250 = int(sfreq_original * 2)  # 250Hz×2s=500个采样点（原始起始点）
# 5.3 生成250Hz下的精准触发点（按官方4.5s间隔，避免超出数据范围）
onset_samples_250 = np.arange(onset_250, onset_250 + total_trial_duration * sfreq_original, 
                              int(trial_interval * sfreq_original), dtype=int)
# 5.4 转换触发点至128Hz（关键：与重采样后的gdf数据严格时间对齐）
onset_samples_128 = np.round(onset_samples_250 * (sfreq / sfreq_original)).astype(int)
# 5.5 边界校验：确保触发点不超出gdf总采样点数（避免后续Epochs报错）
onset_samples_128 = onset_samples_128[onset_samples_128 < total_samples_128 - int(sfreq*5)]
print("最终生成触发点数：", len(onset_samples_128))  # 验证：288

# ===================== 步骤6：构造MNE标准事件数组（强制格式：n×3） =====================
# MNE事件数组要求：[[采样点, 0, 事件码], [采样点, 0, 事件码], ...]
events = np.column_stack([onset_samples_128, np.zeros_like(event_codes), event_codes])
event_id = {'left':769, 'right':770, 'feet':771, 'tongue':772}  # 事件码-任务映射
print("构造MNE标准事件数：", len(events))  # 验证：288

# ===================== 步骤7：提取试次数据+通道内Z-score标准化（必备！） =====================
epochs = mne.Epochs(
    raw, events, event_id=event_id,
    tmin=-0.5, tmax=4.0,  # 截取：刺激前0.5s（基线）→ 刺激后4.0s，共4.5s
    baseline=(-0.5, 0),   # 基线校正：用-0.5s~0s的信号校正试次数据
    preload=True, verbose=False, on_missing='ignore'
)
# 提取数据并标准化（通道内Z-score，消除幅值差异，让模型聚焦有效特征）
X = epochs.get_data().astype(np.float32)  # 形状：(288,22,577)
X = (X - np.mean(X, axis=2, keepdims=True)) / (np.std(X, axis=2, keepdims=True) + 1e-8)
# 标签归一化（769→0,770→1,771→2,772→3），适配模型独热编码
y = epochs.events[:, 2]
y = np.array([[769,770,771,772].index(code) for code in y])
print("试次数据形状：", X.shape, "标签形状：", y.shape)  # 验证：(288,22,577) (288,)

# ===================== 步骤8：EEG小样本数据增强（轻量无失真，缓解过拟合） =====================
def eeg_data_augment(X, y, aug_times=1):
    """EEG专用增强：时间轴±1~2采样点平移，不改变有效特征"""
    X_aug = X.copy()
    y_aug = y.copy()
    n_trials, Chans, Samples = X.shape
    for _ in range(aug_times):
        shift = np.random.randint(-2, 3, size=n_trials)
        X_temp = np.zeros_like(X)
        for i in range(n_trials):
            if shift[i] > 0:
                X_temp[i, :, shift[i]:] = X[i, :, :-shift[i]]
            elif shift[i] < 0:
                X_temp[i, :, :shift[i]] = X[i, :, -shift[i]:]
            else:
                X_temp[i] = X[i]
        X_aug = np.concatenate([X_aug, X_temp], axis=0)
        y_aug = np.concatenate([y_aug, y], axis=0)
    return X_aug, y_aug

# 增强2倍（288→576样本，小样本适配，不引入噪声）
X_aug, y_aug = eeg_data_augment(X, y, aug_times=2)  
print("数据增强后形状：", X_aug.shape, "标签形状：", y_aug.shape)  # 验证：(864,22,577) (864,)

# ===================== 步骤9：分层划分训练/验证/测试集（小样本专用，避免类别失衡） =====================
# 8:1:1划分，分层保证4类样本比例一致
X_train, X_temp, y_train, y_temp = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
# 独热编码
y_train_one_hot = to_categorical(y_train, num_classes=4)
y_val_one_hot = to_categorical(y_val, num_classes=4)
y_test_one_hot = to_categorical(y_test, num_classes=4)
print("训练集：", X_train.shape, "验证集：", X_val.shape, "测试集：", X_test.shape)

# ===================== 步骤10：EEGNet模型初始化（适配22通道+128Hz） =====================
Chans = X_train.shape[1]
Samples = X_train.shape[2]
# 模型初始化参数修改
model = EEGNet(
    nb_classes=4,
    Chans=Chans,
    Samples=Samples,
    dropoutRate=0.2,  # 进一步降低Dropout（从0.3→0.2），保留更多特征
    kernLength=16,    # 缩短时间卷积核（从32→16），适配128Hz下0.125s精细时间特征
    F1=24,            # 增加卷积核数（从16→24），强化特征提取能力
    D=2,
    F2=48,            # F2=F1×D，遵循原论文
    norm_rate=0.05,   # 降低归一化系数（从0.1→0.05），增强模型拟合能力
    dropoutType='Dropout'
)

# 优化器学习率修改（降低学习率，减缓收敛速度，避免震荡）
optimizer = Adam(learning_rate=5e-4, decay=5e-6)  # 1e-3→5e-4，1e-5→5e-6
model.summary()

# ===================== 步骤11：模型编译+训练（小样本优化配置，提升泛化） =====================
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# 早停机制：监控验证集准确率，保存泛化最好的模型
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)
log_dir = f"bci_tensorboard_logs/{time.strftime('%Y%m%d_%H%M%S')}"  # 时间戳日志目录
tensorboard_callback = TensorBoard(
    log_dir=log_dir,        # 日志保存路径（自动创建文件夹）
    histogram_freq=0,       # 关闭直方图（仅看准确率，提速减日志体积）
    write_graph=False,      # 关闭计算图（精简日志，TF2.3.4推荐）
    write_images=False,     # 关闭权重保存，避免日志过大
    update_freq='epoch',    # 按轮次更新，每轮记录1次准确率（训练集+验证集）
    profile_batch=0         # 关闭性能分析（TF2.3.4需显式设0，否则会报警告）
)
# 开始训练
print("\n开始训练...")
history = model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_val, y_val_one_hot),
    batch_size=16,
    epochs=100,
    shuffle=True,
    verbose=1,
    callbacks=[early_stopping, tensorboard_callback]
)
os.makedirs('saved_models', exist_ok=True)
model_name = f"bci2a_simple_mat_eegnet_best_model_{time.strftime('%Y%m%d_%H%M%S')}.h5"
model_save_path = f'saved_models\\{model_name}'
model.save(model_save_path)
print("模型训练完成并保存！")

# ===================== 步骤12：模型最终评估（测试集） =====================
print("\n开始模型评估...")
val_loss, val_acc = model.evaluate(X_val, y_val_one_hot, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=0)
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)
sk_acc = accuracy_score(y_true, y_pred)

print(f"验证集准确率：{val_acc:.4f}（{val_acc*100:.2f}%）")
print(f"测试集准确率（model.evaluate）：{test_acc:.4f}（{test_acc*100:.2f}%）")
print(f"测试集准确率（sklearn）：{sk_acc:.4f}（{sk_acc*100:.2f}%）")