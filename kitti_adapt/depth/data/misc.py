from torch.utils.data import Subset, ConcatDataset


def mix_datasets(real_dataset, virtual_dataset, ratio):
    """
    Create a ratio based mixed; example ratio = 0.75 would mean that the data consists of 75 % real dataset and 25 % virtual dataset
    """
    num_real = int(ratio * len(real_dataset))
    num_real_discard = len(real_dataset) - num_real
    num_virtual = int((1 - ratio) * len(virtual_dataset))
    num_virtual_discard = len(virtual_dataset) - num_virtual
    half1, _ = random_split(real_dataset, [num_real, num_real_discard])
    half2, _ = random_split(virtual_dataset, [num_virtual, num_virtual_discard])
    mixed = ConcatDataset([half1, half2])
    return mixed
