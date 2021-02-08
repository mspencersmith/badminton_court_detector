"""
Load model for evaluation here. You can request a processor but if cuda is 
unavailable then model will be loaded to cpu

"""
import torch
from nn_model import Net

def load_model(model, processor):
    """Loads model on to GPU if availible otherwise loads to CPU"""   
    if "cuda" in processor and torch.cuda.is_available()==False:
        print("\nCuda unavailable\n")
        processor = "cpu"
    print(f"\nLoading neural network on {processor}..\n")
    device = torch.device(processor)
    net = Net().to(device)
    net.load_state_dict(torch.load(model, map_location=device))
    net.eval()
    return device, net
