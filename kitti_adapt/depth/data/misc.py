from torch.utils.data import Subset, ConcatDataset, DataLoader, random_split
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F


def mix_datasets(real_dataset, virtual_dataset, real_fraction, virtual_fraction):
    num_real = int(round(real_fraction * len(real_dataset)))
    num_virtual = int(round(virtual_fraction * len(virtual_dataset)))
    print(f"Using {num_real}({real_fraction * 100}%) real samples and {num_virtual}({virtual_fraction * 100}%) virtual samples.")
    real_subset = Subset(real_dataset, list(range(num_real)))
    virtual_subset = Subset(virtual_dataset, list(range(num_virtual)))
    mixed = ConcatDataset([real_subset, virtual_subset])
    return mixed


def model_depth_output_to_numpy(depth_tensor, as_pil=False):
        depth_tensor = depth_tensor[0, 0] * 65535
        depth_tensor = depth_tensor.cpu().detach().numpy().round().astype(np.uint32)
        depth = Image.fromarray(depth_tensor) if as_pil else depth_tensor
        return depth


def compute_errors(gt, pred):
    # Source: https://github.com/mrharicot/monodepth
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt + 1) - np.log(pred + 1)) ** 2))
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_model(model, dataset, device=None, update_stats_every=10, limit=None):
    device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    model.to(device)
    METRIC_NAMES = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

    num_samples = len(dataset)
    if limit:
        num_samples = min(limit, num_samples)

    all_errors = []
    with tqdm(total=num_samples) as pbar:
        for i in range(num_samples):
            sample = dataset.get(i, transform=False)
            depth = sample["depth"]

            sample_tensor = dataset[i]
            rgb_tensor = sample_tensor["rgb"]
            depth_tensor = sample_tensor["depth"]

            predicted_depth_tensor = model(rgb_tensor.unsqueeze(0).cuda())
            predicted_depth_tensor = F.interpolate(predicted_depth_tensor, size=depth.size[::-1])
            predicted_depth = model_depth_output_to_numpy(predicted_depth_tensor, as_pil=False)

            errors = compute_errors(np.array(depth), predicted_depth)
            all_errors.append(errors)

            if i % update_stats_every == 0:
                postfix = dict(zip(METRIC_NAMES, errors))
                pbar.set_postfix(**postfix)
            pbar.update(1)

    all_errors = np.array(all_errors)
    mean_metrics = all_errors.mean(axis=0).tolist()
    return dict(zip(METRIC_NAMES, mean_metrics))
