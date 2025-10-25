import numpy as np
from bci_predictor import BCIPredictor

def test_interface():
    predictor = BCIPredictor()
    
    predictor.load_model(
        model_path='trained_model_weights.pth',
        n_channels=30,
        n_times=1152,
        n_classes=5,
        sampling_rate=256
    )
    print("model load successfully!")

    n_trials = 3
    n_channels = 30
    n_times = 1152
        
    test_data = np.random.randn(n_trials, n_channels, n_times) * 0.1

    predictions = predictor.predict(test_data)
    print(f"batch predict successfully: {predictions.shape}")

    single_result = predictor.predict_single(test_data[0])
    print(f"predict successfully! class:{single_result['predicted_class']}, confidence:{single_result['confidence']:.3f}")
    model_info = predictor.get_model_info()
    print(f"model information: {model_info}")
        
    print("\ntests pass!")

if __name__ == "__main__":
    test_interface()
