import torch


def load_model():
    # TODO parametrize the model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model
