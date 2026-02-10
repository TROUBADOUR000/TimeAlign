import numpy as np
from numpy.fft import fft, ifft

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    hfmse = HF_MSE(pred, true)
    hfmae = HF_MAE(pred, true)

    return mae, mse, hfmse, hfmae, rmse, mape, mspe


def HF_MSE(pred, true, hf_start_ratio=0.4):
    """
    Calculates High-Frequency Mean Squared Error (HF-MSE) in the frequency domain.
    The high-frequency band is defined from hf_start_ratio * N up to N/2 (and symmetrically).
    
    Args:
        pred (np.ndarray): Predicted time series.
        true (np.ndarray): True time series.
        hf_start_ratio (float): Ratio defining the start of the high-frequency band (e.g., 0.4).
        
    Returns:
        float: HF-MSE value (mean squared magnitude difference).
    """
    num_time_steps = len(true)
    
    # 1. 傅里叶变换
    true_freq = fft(true)
    pred_freq = fft(pred)
    diff_freq = true_freq - pred_freq
    
    # 2. 确定高频索引并隔离分量
    hf_start_idx = int(num_time_steps * hf_start_ratio)
    # 截止到 Nyquist 频率 (正频率部分)
    hf_mid_idx = int(num_time_steps / 2) + 1 
    
    hf_diff_freq = np.zeros_like(diff_freq, dtype=complex)
    
    # 隔离正频率部分
    hf_diff_freq[hf_start_idx:hf_mid_idx] = diff_freq[hf_start_idx:hf_mid_idx]
    
    # 隔离负频率部分 (对称部分)
    hf_diff_freq[num_time_steps - hf_mid_idx : num_time_steps - hf_start_idx] = \
        diff_freq[num_time_steps - hf_mid_idx : num_time_steps - hf_start_idx]
            
    # 统计有效高频分量的数量
    T_hf = np.count_nonzero(hf_diff_freq)
    if T_hf == 0:
        return 0.0

    # 3. 计算 HF-MSE (频域差值平方模长的均值)
    hf_mse_value = np.sum(np.abs(hf_diff_freq) ** 2) / T_hf
    
    return hf_mse_value

def HF_MAE(pred, true, hf_start_ratio=0.4):
    """
    Calculates High-Frequency Mean Absolute Error (HF-MAE) in the frequency domain.
    The high-frequency band is defined from hf_start_ratio * N up to N/2 (and symmetrically).
    
    Args:
        pred (np.ndarray): Predicted time series.
        true (np.ndarray): True time series.
        hf_start_ratio (float): Ratio defining the start of the high-frequency band (e.g., 0.4).
        
    Returns:
        float: HF-MAE value (mean absolute magnitude difference).
    """
    num_time_steps = len(true)
    
    # 1. 傅里叶变换
    true_freq = fft(true)
    pred_freq = fft(pred)
    diff_freq = true_freq - pred_freq
    
    # 2. 确定高频索引并隔离分量
    hf_start_idx = int(num_time_steps * hf_start_ratio)
    hf_mid_idx = int(num_time_steps / 2) + 1
    
    hf_diff_freq = np.zeros_like(diff_freq, dtype=complex)
    
    # 隔离正频率部分
    hf_diff_freq[hf_start_idx:hf_mid_idx] = diff_freq[hf_start_idx:hf_mid_idx]
    
    # 隔离负频率部分 (对称部分)
    hf_diff_freq[num_time_steps - hf_mid_idx : num_time_steps - hf_start_idx] = \
        diff_freq[num_time_steps - hf_mid_idx : num_time_steps - hf_start_idx]
            
    # 统计有效高频分量的数量
    T_hf = np.count_nonzero(hf_diff_freq)
    if T_hf == 0:
        return 0.0

    # 3. 计算 HF-MAE (频域差值模长的均值)
    hf_mae_value = np.sum(np.abs(hf_diff_freq)) / T_hf
    
    return hf_mae_value