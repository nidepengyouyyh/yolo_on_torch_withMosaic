import torch
from models.model import Detector

def load_weights_with_mapping(model, weight_path):
    # 加载权重文件
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # # 创建一个新的字典用于存储映射后的权重
    # new_state_dict = {}
    # for model_key in models.state_dict().keys():
    #     # 获取权重文件中对应的键名
    #     # checkpoint_key = model_key.replace('stem.', 'backbone.stem.')
    #
    #     if model_key in model_weights:
    #         new_state_dict[model_key] = model_weights[model_key]
    #     else:
    #         print(f"{model_key}: Not found in weight file.")

    # 加载映射后的权重
    model.load_state_dict(model_weights, strict=False)
    return model

def select_weights_with_mapping(model, weight_path):
    # 加载权重文件
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    selected_layers = extract_feature(model)
    # 创建一个新的字典用于存储选择后的权重
    selected_state_dict = {}
    for model_key in model.state_dict().keys():
        # 检查是否为选择的层，并且存在于加载的权重文件中
        if any(layer in model_key for layer in selected_layers) and model_key in model_weights:
            selected_state_dict[model_key] = model_weights[model_key]
        else:
            print(f"{model_key}: Ignored or not found in weight file.")

    # 加载选择后的权重
    model.load_state_dict(selected_state_dict, strict=False)
    return model

def print_state_dict_keys(model):
    print("Model state_dict keys:")
    for key in model.state_dict().keys():
        print(key)

def extract_feature(model):
    load_list = []
    for key in model.state_dict().keys():
        if "backbone" in key or "neck" in key:
            load_list.append(key)
    return load_list

# if __name__ == '__main__':
#     detector = Detector(num_class=15)
#     ckp_dir = './ckp/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'
#     model = select_weights_with_mapping(detector, ckp_dir)
