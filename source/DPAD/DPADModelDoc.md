# DPADModel formulation
The model learning for `DPADModel` is done in 4 steps as follows ([**Methods**](https://doi.org/10.1038/s41593-024-01731-2)): 

1. In the first optimization step, we learn the parameters $A'^{(1)}(\cdot)$, $K^{(1)}(\cdot)$, and $C^{(1)}_z(\cdot)$ of the following RNN:

$$x^{(1)}_{k+1} = A'^{(1)}(x^{(1)}_k) + K^{(1)}( y_k )$$

$$\hat{z}^{(1)}_k = C_z^{(1)}( x^{(1)}_k )$$

and estimate its latent state $x^{(1)}_k\in\mathbb{R}^{n_1}$, while minimizing the negative log-likelihood (NLL) of predicting the behavior $z_k$ as $\hat{z}^{(1)}_k$. This RNN is implemented as an `RNNModel` object with $y_k$ as the input and $\hat{z}^{(1)}_k$ as the output and the state dimension of $n_1$ as specified by the user. `RNNModel` implements each of the RNN parameters, $A'^{(1)}(\cdot)$, $K^{(1)}(\cdot)$, and $C^{(1)}_z(\cdot)$, as a multilayer feed-forward neural network implemented via the `RegressionModel` class. 


2. The second optimization step uses the extracted latent state $x^{(1)}_k$ from the RNN and fits the parameter $C_y^{(1)}$ in

$$\hat{y}_k = C_y^{(1)}( x^{(1)}_k )$$

while minimizing the NLL of predicting the neural activity $y_k$ as $\hat{y}_k$. The $C_y^{(1)}$ parameter that specifies this mapping is implemented as a flexible multilayer feed-forward neural network, via the `RegressionModel` class. 


3. In the third optimization step, we learn the parameters $A^{(2)}(\cdot)$, $K^{(2)}(\cdot)$, and $C^{(2)}_y(\cdot)$ of the following RNN:  

$$x^{(2)}_{k+1} = A'^{(2)}(x^{(2)}_k) + K^{(2)}( y_k, x^{(1)}_{k+1} )$$

$$\hat{y}_k = C_y^{(2)}( x^{(2)}_k )$$

and estimate its latent state $x^{(2)}_k$ while minimizing the aggregate neural prediction negative log-likelihood, which also takes into account the negative log-likelihood (NLL) obtained from step 2 via the $C_y^{(1)}( x^{(1)}_k )$ and computed using the previously learned parameter $C_y^{(1)}$ and the previously extracted states $x_k^{(1)}$ in steps 1-2. This RNN is also implemented as an `RNNModel` object with the concatenation of $y_k$ and $x^{(1)}_k$ as the input and the predicted neural activity as the output. The NLL for predicting neural activity from steps 1-2 is also provided as input, to allow formation of aggregate neural prediction NLL as the loss. `RNNModel` again implements each of the RNN parameters, $A'^{(2)}(\cdot)$, $K^{(2)}(\cdot)$, and $C^{(2)}_y(\cdot)$, as a multilayer feed-forward neural network implemented via the `RegressionModel` class. 


4. The fourth optimization step uses the extracted latent states in optimization steps 1 and 3 (i.e., $x^{(1)}_k$ and $x^{(2)}_k$) and fits $C_z$ in:

$$\hat{z}_k = C_z( x^{(1)}_k, x^{(2)}_k )$$

while minimizing the behavior prediction negative log-likelihood. This step again implements $C_z(.)$ as a flexible multilayer feed-forward neural network, via the `RegressionModel` class.

For additional options and generalizations to these steps, please read **Methods** in the [DPAD paper](https://doi.org/10.1038/s41593-024-01731-2).

# Objective function of each optimization step
Objective function of each optimization step is the negative log-likelihood (NLL) associated with the time series predicted in that optimization step, i.e. $z_k$ for steps 1 and 4 and $y_k$ for steps 2 and 3. 
For Gaussian distributed signals $z_k$ with isotropic noise, the NLL is proportional to the mean squared errors (MSE). For example, for Gaussian behaviors loss of step 1 will be: 

$$\sum_{k}\Vert z_k-\hat{z}^{(1)}_k\Vert_2^2$$

To support non-Gaussian data modalities, e.g., categorical behavior, DPAD adjusts the objectives of the four optimization steps and the architecture of the readout parameters based on the NLL of the relevant distribution. For example, for categorical behavior $z_k$ the NLL is proportional to the cross-entropy and the readout architecture is adjusted as follows: 
1) we change the behavior readout parameter $C_z$ to have an output dimension of $n_z \times n_c$ instead of $n_z$, where $n_c$ denotes the number of behavior categories or classes, and 
2) we apply a Softmax normalization to the output of the behavior readout parameter $C_z$ to ensure that for each of the $n_z$ behavior dimensions, the predicted probabilities for all the $n_c$ classes add up to 1, so that they represent valid probability mass functions. 

For details, see [**Methods**](https://doi.org/10.1038/s41593-024-01731-2).

We also extend DPAD to modeling intermittently measured behavior time series. To do so, when forming the behavior loss, we only compute the NLL loss on samples where the behavior is measured (i.e., mask the other samples) and solve the optimization with this loss. Doing so, the modeling approach becomes applicable to intermittently measured behavior signals (**ED Figs. 8-9, S Fig. 8** in the [DPAD paper](https://doi.org/10.1038/s41593-024-01731-2)).

