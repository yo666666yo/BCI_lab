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

    def predict(self, eeg_data: np.ndarray) -> np.ndarray:
        return self._model.predict(eeg_data)
    
    def predict_single(self, eeg_trial: np.ndarray) -> dict:
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
        return self._config.copy()
    
    def is_ready(self) -> bool:
        return self._is_loaded

def create_predictor(model_path: str, **kwargs) -> BCIPredictor:
    predictor = BCIPredictor()
    predictor.load_model(model_path, **kwargs)
    return predictor
