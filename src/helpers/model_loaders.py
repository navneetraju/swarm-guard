import torch

from src.modules.graph_encoder import UPFDGraphSageNet


def load_pre_trained_graph_encoder(model_path: str, device: str = "cpu") -> UPFDGraphSageNet:
    print(f"Loading graph encoder from {model_path}")
    model_file = torch.load(model_path)
    state_dict = model_file['model_state_dict']
    config = model_file['config']
    model = UPFDGraphSageNet(
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        num_classes=config['num_classes'],
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("Graph encoder loaded successfully on device:", device)
    return model
