import torchmetrics
from datalaoder.LinearDataLoader import Dataset
import torch

import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


ort_session = onnxruntime.InferenceSession(r'./best_loss.onnx')

dataset = Dataset(r'./data_custom_ml/test_dataset.hdf5', normalize=False)
train_test_split = int(len(dataset) * 0.3)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset,
    [train_test_split, len(dataset) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=1,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=1,
    shuffle=True
)

metric_test = torchmetrics.Accuracy()
i = 0
for x, y in dataloader_test:
    print(x.shape)
    x = x.float()
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    metric_test(torch.from_numpy(img_out_y), y)
    if i%64 == 0:
        print(torch.from_numpy(img_out_y), y, metric_test.compute())


# compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)
