"""
Load model for evaluation here. You can request a processor but if cuda is 
unavailable then model will be loaded to cpu

"""
import torch
from nn_model import Net

class LoadModel:
    """Loads model"""
    def __init__(self, model, processor):
        """Initialises model name and processor"""
        self.model = model
        self.processor = processor
        self.load_model()

    def load_model(self):
        """Loads model on to GPU if availible otherwise loads to CPU"""   
        if "cuda" in self.processor and torch.cuda.is_available()==False:
            print("\nCuda unavailable\n")
            self.processor = "cpu"
        print(f"\nLoading neural network on {self.processor}..\n")
        self.device, self.net = self._to_dev()
        
    def _to_dev(self):
        """Initialises model onto device"""
        device = torch.device(self.processor)
        net = Net().to(device)
        net.load_state_dict(torch.load(self.model, map_location=device))
        net.eval()
        return device, net
    