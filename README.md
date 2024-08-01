# Low-Rank-Approximation
Explore low-rank approximation in fully connected neural networks

In this notebook, you're going to explore trained neural networks, and study the rank of its matrices.

**Reminder**: The rank is the number of independent columns of the matrix. If a matrix $A \in \mathbb{R}^{n\times m}$  has rank $k$, then $A$ can be approximated by

$$A \approx B \cdot C$$

where $B \in \mathbb{R}^{n\times k}$ and $C \in \mathbb{R}^{k\times m}$.

You can find the rank of matrix $A$ by performing Gaussian elimination and counting the number of pivots. This can be done in few lines of `numpy` code.

**References**:
- https://arxiv.org/pdf/1804.08838
- https://arxiv.org/pdf/2209.13569
- https://arxiv.org/pdf/2012.13255

Note: The references above are not needed to complete this notebook, but reading them might give you additional insights.

## Important

1. For all the training done, make sure to plot things like the loss values and accuracy on each epoch.

    - You can either use tensorboard or just make a static matplotlib plot.
    
2. Don't add biases to the layers in the network, not important for this notebook.
3. No need to use Dropout or BatchNorm on the network.
4. Remember to use GPUs during the training.
5. Always test your hypothesis on both training and testing sets, you might get a surprising result sometimes.
