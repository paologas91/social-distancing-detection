import torch


def load_model(model):
    """
    Loads the specific Yolo model
    :param model: the model to load
    :return:
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5' + model, pretrained=True, verbose=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model
