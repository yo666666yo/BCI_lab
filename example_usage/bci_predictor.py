import torch
import numpy as np
from typing import Union, List, Optional
import warnings
import os
warnings.filterwarnings('ignore')

class BCIPredictor:
    def __init__(self):
        self._model = None
        self._is_loaded = False
        self._config = {}
        
    def load_model(self, 
                  model_path: str,
                  n_channels: int = 19,
                  n_times: int = 256, 
                  n_classes: int = 4,
                  sampling_rate: int = 128):
        try:
            from implementation import HiddenBCIModel
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"model does not exist: {model_path}")
            self._model = HiddenBCIModel(
                n_channels=n_channels,
                n_times=n_times,
                n_classes=n_classes,
                sampling_rate=sampling_rate
            )
            self._model.load_weights(model_path)
            
            self._is_loaded = True
            self._config = {
                'n_channels': n_channels,
                'n_times': n_times,
                'n_classes': n_classes,
                'sampling_rate': sampling_rate
            }
            
            print(f"model load successfully: {n_channels}channel, {n_classes}class")
            
        except Exception as e:
            raise RuntimeError(f"model load fail: {str(e)}")
    
    def predict(self, eeg_data: np.ndarray) -> np.ndarray:
        if not self._is_loaded:
            raise RuntimeError("please load model")
        if len(eeg_data.shape) != 3:
            raise ValueError(f"imput should be 3-D array: {eeg_data.shape}")
        
        n_trials, n_channels, n_times = eeg_data.shape
        
        if n_channels != self._config['n_channels']:
            raise ValueError(f"channel mismatch: expect{self._config['n_channels']}, got{n_channels}")
        
        if n_times != self._config['n_times']:
            print(f"time point mismatch: expect{self._config['n_times']}, got{n_times}")
        
        return self._model.predict(eeg_data)
    
    def predict_single(self, eeg_trial: np.ndarray) -> dict:
        if not self._is_loaded:
            raise RuntimeError("please load model")
        eeg_data = np.expand_dims(eeg_trial, axis=0)
        predictions = self.predict(eeg_data)
        pred_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return {
            'predicted_class': int(pred_class),
            'confidence': float(confidence),
            'probabilities': predictions[0].tolist()
        }
    
    def get_model_info(self) -> dict:
        if not self._is_loaded:
            raise RuntimeError("model load fail")
        return self._config.copy()
    
    def is_ready(self) -> bool:
        return self._is_loaded

def create_predictor(model_path: str, **kwargs) -> BCIPredictor:
    predictor = BCIPredictor()
    predictor.load_model(model_path, **kwargs)
    return predictor
