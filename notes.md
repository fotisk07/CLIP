# CLIP reproduction (Implementing a paper without crying)

## First Steps

### Baseline

We will first create a benchmark to compare our models against. I choose the simplest model,
a one layer MLP. It performs remarquably well (scarily so) at about 0.5 cross entropy.

Also add a little script accuracy calculation at the end for sanity checks. This super simple model achieves 90% accuracy.
Did a sanity check and at init the model displays 10% accuracy which is expected of a randomly initialised model.

### Simplest clip model
