# RNNModel formulation, 1-step ahead, no input
The formulation for `RNNModel` is as follows  

$$
x_{k+1} = A(x_{k}) + K( y_k ) \\
z_k = C( x_k )
$$  

where $y_k$ and $z_k$ are the input and output, respectively and $x_k$ is the latent state. Each parameter ($A(.)$, $K(.)$, and $C(.)$) is a multi-layer perceptron (MLP) implemented via the `RegressionModel` class.

In the special case of a linear model with the same output time series as the input time series (from $y_k$ to $y_k$), this reduces to a Kalman filter doing one-step ahead prediction:  

$$
x_{k+1} = A x_k + K y_k \\
y_k = C x_k + e_k
$$  

where $x_k \triangleq x_{k|k-1}$ is the estimated latent state at time step $k$ given all inputs up to time step $k-1$. 

Read more on the links to the linear case in **S Note 1** in the [DPAD paper](https://doi.org/10.1038/s41593-024-01731-2).