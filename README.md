# Bayesian Optimization for Parameter Tuning of the XOR Neural Network - California Insitute of Technology



The following Project investigates the use of Bayesian Optimzation to tune Parameters of the XOR Neural Network see below:


<img src="https://github.com/LawrenceMMStewart/Bayesian_Optimization/blob/master/Images/XOR.png" width="450">

Where <img src="https://github.com/LawrenceMMStewart/Bayesian_Optimization/blob/master/Images/inputnode.png" width="30"> is an input node
<img src="https://github.com/LawrenceMMStewart/Bayesian_Optimization/blob/master/Images/sigmoidnode.png" width="30"> is a sigmoid node with hyper-parameter (scale factor) h. 


The research paper "Bayesian Optimization for Parameter Tuning of the XOR Neural Network" can be found on arXiv at: https://arxiv.org/pdf/1709.07842.pdf





### Installing

First ensure that you have julia installed:

```
pip install julia 
```
If you have troubleshooting problems julia is available from https://julialang.org/downloads/. 

Once julia will need to install the official Julia distributions package (for more information see https://github.com/JuliaStats/Distributions.jl. 

```
Pkg.add("Distributions.jl")
```
With both of the above installed, the final step is downloading the repository:

```
Pkg.clone("git://github.com/LawrenceMMStewart/Bayesian_Optimization")
```






# Contents 

* **Kernals.jl** - A selection of kernal functions for Gaussian Processes
* **gaussian_process.jl** - Code for a Gaussian Process (Niave) and a faster Cholesky method 
* **XOR.jl** - A simple XOR neural network (for the 1D classic learning rate problem)
* **XOR_MD.jl** - Generalised XOR with 2 dimensions (alpha and theta)
* **Tune_MD.jl** - Bayesian Optimization with LCB utility function to tune the XOR NN over P
* **Grid_vs_Bayes_XOR.jl** - Compares Random Grid Search with Bayesian Optimization
* **XOR_Timings.jl** -Timings as found in the paper listed above
* **Metric_Strength_Test.jl** -Experiment to see how strong the LCB metric performed for this NN (see paper)


## Future Work 

* Gaussian Processes can be made to O(nlog^2(n)) [3]
* Utility function can be updated to IMGPO  [5] 
* In the folder Builds/RNN's/Examples are the begginnigs of applying Bayesian Optimization to a reccurent neural net used to model the temporal XOR sequence. Hopefully this will be continued at some point in the near future.






## Papers used for code

Code utilizes theory from the following papers:

[1] Jonas Mockus. Application of bayesian approach to numerical methods of global and stochastic optimization. Journal of Global Optimization, 4(4):347– 365, 1994.

[2] Eric Brochu, Vlad M. Cora, and Nando de Freitas. A tutorial on bayesian optimization of expensive cost functions, with application to active user mod- eling and hierarchical reinforcement learning. CoRR, abs/1012.2599, 2010.

[3] Carl Edward Rasmussen and Christopher KI Williams. Gaussian processes for machine learning, volume 1. MIT press Cambridge, 2006.

[4] Sivaram Ambikasaran, Daniel Foreman-Mackey, Leslie Greengard, David W Hogg, and Michael O’Neil. Fast direct methods for gaussian processes. arXiv preprint arXiv:1403.6015, 2014.

[5] Kenji Kawaguchi, Leslie Pack Kaelbling, and Tom ́as Lozano-P ́erez. Bayesian optimization with exponential convergence. In Advances in Neural Informa- tion Processing Systems, pages 2809–2817, 2015.

[6] Gerhard J. Woeginger Alexander S. Kulikov. Computer science - theory and applications. 11th International Computer Science Symposium in Russia, CSR, 2016.

[7] Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179–211, 1990.

[8] Robert A Jacobs. Increased rates of convergence through learning rate adap- tation. Neural networks, 1(4):295–307, 1988.

[9] Federico Girosi, Michael Jones, and Tomaso Poggio. Regularization theory and neural networks architectures. Neural computation, 7(2):219–269, 1995.

[10] Ziyu Wang, Frank Hutter, Masrour Zoghi, David Matheson, and Nando de Feitas. Bayesian optimization in a billion dimensions via random em- beddings. Journal of Artificial Intelligence Research, 55:361–387, 2016.
8
[11] Jasper Snoek, Oren Rippel, Kevin Swersky, Ryan Kiros, Nadathur Satish, Narayanan Sundaram, Mostofa Patwary, Mr Prabhat, and Ryan Adams. Scal- able bayesian optimization using deep neural networks. In International Con- ference on Machine Learning, pages 2171–2180, 2015.



## License

The code is distributed under a Creative Commons Attribution 4.0 International Public License. If you use this code please attribute to L. Stewart and M.A. Stalzer Bayesian Optimization for Parameter Tuning of the XOR Neural Network, 2017.

## Acknowledgments

This research is funded by the Caltech SURF program and the Gordon and Betty Moore Foundation through Grant GBMF4915 to the Caltech Center for Data-Driven Discovery. Many thanks to Mark Stalzer for supervising this project.

