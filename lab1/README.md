# Notes

## Quickstart

```bash
python -m src.transformer.train
```

## Transformer

- We want to train the transformer on all block sizes from 1 to n, so the it can learn to operate on all sizes not only the the max block size
- We process `mini batches` of text to keep the GPU busy and levrege its parallerization capabilities
- When using the GPU we have to move every "cooperating" elements to it: model.to(device), x, y (targets).
- `@torch.no_grad()` for function to make the operations inside them ignored by the `.backward()`. Makes the operations much more efficent.
- if `wei` (weights) have very large variance (large or small numbers) the softmax will converge to the one-hot vector. That's why we divide by the sqrt of the head_size. It would result in aggregating information mostly form the single node (the largest number).
- typically biases are not used inside single head attention modules
- `key`, `query` and `value` matrixes are `linear projections`
- when something is not the parameter in the nn.Module, in torch conventions it should be registered as a buffer (i.e. trill)
