# Changes 
Versioning follows [semver](https://semver.org/).

- v0.0.9:
  - Changes the default Early Stopping setting to be based on validation loss.
  - Makes the metric computation in inner cross validation immune to flat channels in the validation data (previously if a neuron was flat in validation data of one inner cross validation fold, the self-prediction for that fold would become NaN).
  - Adds Gaussian Smoother tool to use in notebooks. 

- v0.0.8:
  - Enables z-scoring of inputs to the model by default. Change `zscore_inputs` to `False` or add `nzs` to `methodCode` to disable.
