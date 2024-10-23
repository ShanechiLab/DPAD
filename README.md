# Publication:
The following paper introduces and provides results of DPAD (**dissociative and prioritized analysis of dynamics**) in multiple real neural datasets.

Omid G. Sani, Bijan Pesaran, Maryam M. Shanechi. *Dissociative and prioritized modeling of behaviorally relevant neural dynamics using recurrent neural networks*. ***Nature Neuroscience*** (2024). https://doi.org/10.1038/s41593-024-01731-2

Original preprint: https://doi.org/10.1101/2021.09.03.458628


# Usage examples
The following notebook contains usage examples of DPAD for several use-cases:
[source/DPAD/example/DPAD_tutorial.ipynb](https://github.dev/ShanechiLab/DPAD/blob/main/source/DPAD/example/DPAD_tutorial.ipynb). 

An HTML version of the notebook is also available next to it in the same directory.

# Usage examples
The following documents explain the formulation of the key classes that are used to implement DPAD (the code for these key classes is also available in the same directory):

- [source/DPAD/DPADModelDoc.md](./source/DPAD/DPADModelDoc.md): The formulation implemented by the `DPADModel` class, which performs the overall 4-step DPAD modeling.

-  [source/DPAD/RNNModelDoc.md](./source/DPAD/RNNModelDoc.md): The formulation implemented by the custom `RNNModel` class, which implements the RNNs that are trained in steps 1 and 3 of DPAD. 

-  [source/DPAD/RegressionModelDoc.md](./source/DPAD/RegressionModelDoc.md): The formulation implemented by the `RegressionModel` class, which `RNNModel` and `DPADModel` both internally use to build the general multilayer feed-forward neural networks that are used to implement each model parameter. 

We are working on various improvements to the DPAD codebase. Stay tuned!

# Change Log
You can see the change log in [ChangeLog.md](./ChangeLog.md)  

# License
Copyright (c) 2024 University of Southern California  
See full notice in [LICENSE.md](./LICENSE.md)  
Omid G. Sani and Maryam M. Shanechi  
Shanechi Lab, University of Southern California  
