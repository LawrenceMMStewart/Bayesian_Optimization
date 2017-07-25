#linear_ker.jl

"""
Compute the kernal, (linear) described 

The kernal can be expressed mathematically as k(x,y)=<x,y>

Arguments
---------
x
    First Argument 
y
    Second Argument 


"""

function linear_ker(x,y)
    if size(x)!==size(y)
        error("Error: Input vectors must have the same dimension and shape")
    end
    return dot(x,y)
end





#std_exp_square_ker.jl

"""
Compute the kernal, exponential squared (standard i.e. no parameters), as described in A Tutorial on Bayesian 
Optimization of Expensive Cost Functions, with Application to Active User Modeling and  Hierarchical Reinforcement 
Learning Eric Brochu, Vlad M. Cora and Nando de Freitas December 14, 2010.

The kernal can be expressed mathematically as k(x,y)=exp(-1/2||x-y||^2)

Arguments
---------
x
    First Argument 
y
    Second Argument 


"""

function std_exp_square_ker(x,y)
    if size(x)!==size(y)
        error("Error: Input vectors must have the same dimension and shape")
    end
    return exp(-0.5*(dot(x-y,x-y)))
end





#hyper_exp_square_ker


"""
Compute the kernal, exponential squared (with parameters), as described in A Tutorial on Bayesian 
Optimization of Expensive Cost Functions, with Application to Active User Modeling and  Hierarchical Reinforcement 
Learning Eric Brochu, Vlad M. Cora and Nando de Freitas December 14, 2010.

The kernal can be expressed mathematically as k(x,y)=exp(-1/2(theta)||x-y||^2)

Arguments
---------
x
    First Argument 
y
    Second Argument 
theta
    Hyper-parameter

"""

function hyper_exp_square_ker(x,y,theta)
    if size(x)!==size(y)
        error("Error: Input vectors must have the same dimension and shape")
    end
    return exp(-0.5*(dot(x-y,x-y))/theta)
end








#matern_ker
"""
Compute the matern kernal, as described in A Tutorial on Bayesian 
Optimization of Expensive Cost Functions, with Application to Active User Modeling and  Hierarchical Reinforcement 
Learning Eric Brochu, Vlad M. Cora and Nando de Freitas December 14, 2010.


Arguments
---------
x
    First Argument 
y
    Second Argument 
h
    Hyper-parameter


"""
function matern_ker(x,y,h)
    if size(x)!==size(y)
        error("Error: Input vectors must have the same dimension and shape")
    end

    

    return ((0.5^(h-1))/(gamma(h)))*(2*sqrt(h)*sqrt(dot(x-y,x-y)))^h*besselj(h,2*sqrt(h)*sqrt(dot(x-y,x-y)))
end







#cov_gen.jl

"""
Create the varience-covarience matrix K, as described in A Tutorial on Bayesian 
Optimization of Expensive Cost Functions, with Application to Active User Modeling and  Hierarchical Reinforcement 
Learning Eric Brochu, Vlad M. Cora and Nando de Freitas December 14, 2010. 

Arguments
---------
Ker
    Kernal function
x
    Dataset


"""


function cov_gen(ker,x,y)
    K=zeros(length(x),length(y))
    
    #loop through values and update
    for i=1:length(x)
        for j=1:length(y)
            K[i,j]=ker(x[i],y[j])   #Note for some reason here the way we have set up K means K[1][2] does 
                                        #not work. We need to use K[1,2]. 
            end
    end
    return K
end







