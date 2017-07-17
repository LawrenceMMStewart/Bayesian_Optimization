#GP_UCB.jl

"""
Returns the values of the mean and varience as well as the distribution (this is returned for ease of iteration) 
after adding a single point [x,f]=D or a set of points D=[x1,f1, x2]. 
Themean and varience functions are as described in A Tutorial on Bayesian Optimization of Expensive Cost 
Functions, with Application to Active User Modeling and  Hierarchical Reinforcement Learning Eric Brochu, 
Vlad M. Cora and Nando de Freitas December 14, 2010. 

Arguments
---------
Ker
    Kernal function
D
    Dataset as tuples of (x,y), where y is f(x)+noise
noise 
    noise value (often) 1e-6
xrange
    range of x points to choose (x axis)


"""


function GP_UCB()
end

