# CLIPScope

## Description of the files

- mining.py
  > Mining OOD candidate labels.

- CLIPScope.py
  > Our method

## Implementation of mining.py

### Choose your own lexical database
> The words need to be saved in a python list format.

### Modify mining.py
> Make sure that your saved words can be correctly read.

### Mine OOD candidate labels
Run

```
python mining.py
```

## Implementation of CLIPScope

### Datasets
> Please refer to [MOS](https://github.com/deeplearning-wisc/large_scale_ood) to prepare the ImageNet-1K, iNaturalist, SUN, Places, and Textures.

### ID Labels
> Please use the ImageNet-1K labels as the ID labels.

### Modify CLIPScope.py
> Please ensure that the datasets and labels are read correctly.

### Implementation
```
python CLIPScope.py
```

## Scaled mini implementation

The mini implementation uses torchvision datasets and writes dataset-specific
mined labels. Run the mining step first:

```
python mini_mining.py --id cifar100
python mini_CLIPScope.py --id cifar100
```

The current workflow reads `sorted_nouns_<dataset>.pt`. The older
`mini_sorted_nouns.pt` and `mini_ood_labels.pt` files are no longer used.
