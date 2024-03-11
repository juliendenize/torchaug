# Transforms tutorial

## How transforms work

Transforms are `nn.Module` classes that when called perform a transformation on a data. The data can either be a **single sample** or a **batch of samples**. However to correctly handle the data the relevant transforms need to know if they should work in batch mode are not. To do so, the relevant transforms 