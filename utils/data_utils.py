import torch


def z_score(data: tuple, view_size):
    normalized_data_dict = {}
    for idx, value in enumerate(data):
        data = (value - torch.mean(value.view(view_size))) / torch.std(value.view(view_size))
        normalized_data_dict[idx] = data
        normalized_data_dict[f'{idx}_std'] = torch.std(value.view(view_size))
        normalized_data_dict[f'{idx}_mean'] = torch.mean(value.view(view_size))
    return normalized_data_dict


def count_parameters(model):
    num_of_parameters = sum(p.numel() for p in model.parameters())
    return num_of_parameters
