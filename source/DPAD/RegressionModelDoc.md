# RegressionModel formulation
`RegressionModel` implements a multilayer feed-forward neural network as follows. Each layer applies the following computation to its input:

$$
h = f(Nx+b)
$$

where $x\in\mathbb{R}^{n_x}$ is the input to the layer, $h\in\mathbb{R}^{n_h}$ is the output, and $f(.)$ is a fixed scalar function (typically nonlinear) applied on each dimension of its input vector. The bias vector $b\in\mathbb{R}^{n_h}$ and the matrix $N\in\mathbb{R}^{n_h\times n_x}$ are learnable parameters of the above feed-forward  neural network layer. `RegressionModel` creates multiple of these layers stacked together (each created using `tf.keras.layers.Dense`), feeding the output of each as the input of the next, until finally returning the output of the last layer as the overall output of the multilayer feed-forward neural network. For example, with 1 hidden layer, the overall formulation will be as follows:

$$
z = M f(Nx + b) + b'
$$

where $z\in\mathbb{R}^{n_z}$ is the overall output of the multilayer feed-forward neural network, and $M\in\mathbb{R}^{n_z\times n_h}$ is a matrix. **S Fig. 1c** in the [DPAD paper](https://doi.org/10.1038/s41593-024-01731-2) depicts the computation graph for this case (not showing the bias vectors $b$ and $b'$). We use rectified linear unit (ReLU) functions as the nonlinearity $f(.)$ for all hidden layers and include a bias term $b$ for all hidden layers in this work ([**Methods**](https://doi.org/10.1038/s41593-024-01731-2)). 

When no hidden layers are used and biases are set to 0, the multilayer feed-forward neural network implemented by `RegressionModel` reduces to the special case of a linear matrix multiplication: 

$$
y = Mx
$$

for which the computation graph is shown in **S Fig. 1b** in the [DPAD paper](https://doi.org/10.1038/s41593-024-01731-2).
