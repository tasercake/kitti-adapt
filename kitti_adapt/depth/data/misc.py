from torch.utils.data import Subset, ConcatDataset


# TODO: Train val test split
def train_val_test_split(dataset, subsets):
    num_samples = len(dataset)
    train_ratio = subsets.get("train", 0)
    val_ratio = subsets.get("val", 0)
    test_ratio = subsets.get("test", 0)
    if train_ratio + val_ratio + test_ratio > 1:
        raise ValueError("Sum of subset fractions exceeds 1!")


# TODO: refactor dataset mixer
def create_mix_ratio_dataset(ratio, real_dataset, virtual_dataset):
    """
    Create a ratio based mixed; example ratio = 0.75 would mean that the data consists of 75 % real dataset and 25 % virtual dataset
    """
    real_half = int(ratio * len(real_dataset))
    real__ = len(real_dataset) - real_half
    virtual_half = int((1 - ratio) * len(virtual_dataset))
    virtual__ = len(virtual_dataset) - virtual_half
    half1, _ = random_split(real_dataset, [real_half, real__])
    half2, _ = random_split(virtual_dataset, [virtual_half, virtual__])
    ratio_mixed = ConcatDataset([half1, half2])
    return ratio_mixed
