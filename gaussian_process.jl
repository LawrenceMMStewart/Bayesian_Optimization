#gaussian_process.jl

"""
Returns the values of the mean and varience as well as the distribution (this is returned for ease of iteration) 
after adding a single point [x,f]=D or a set of points D=[x1,f1, x2]. Note this is a very inefficient way of 
computing the gaussian_process as it requires generating the variance covarience matrix for all possible 
selection values. Instead use the cholesky version found below.
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


function gaussian_process(ker,D,noise,xrange)
    K_ss=cov_gen(ker,xrange,xrange)+eye(length(xrange))*noise
    y=map(x->x[2],D)   # these are our y noisy functions
    x=map(x->x[1],D) #These are our x training points
    K = cov_gen(ker,x,x)+noise*eye(length(x))
    K_s=cov_gen(ker,x,xrange)  #K_s is as in nandos code but in the paper called little k
    Inv_K=inv(K)  
    µ=transpose(K_s)*Inv_K*y
    sigma2=diag(K_ss-transpose(K_s)*Inv_K*K_s)  
    sigma=sqrt(sigma2) 
    return (µ,sigma,D)
end
    
    
#Another temporary one:


    
#gaussian_process_chol.jl

"""
Computes the gaussian process of a given set of points xrange with given noise
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

function gaussian_process_chol(ker,D,noise,xrange)
    
    
    y=map(x->x[2],D)   # these are our y noisy functions
    x=map(x->x[1],D) #These are our x training points
    K = cov_gen(ker,x,x)+noise*eye(length(x))
    L=ctranspose(chol(K))
    # temp=cov_gen(ker,x,xrange)
    Lk=\(L,cov_gen(ker,x,xrange)) 

    mu=[dot(Lk[:,i],(\(L,y))) for i=1:size(Lk)[2] ] 
    K_diag=diag_cov_gen(ker,xrange)+ones(length(xrange))*noise
    s2=K_diag-[ sum( (Lk.*Lk)[:,i] ) for i=1:size(Lk)[2] ]  
    sigma=sqrt(s2) 
    return (mu,sigma,D)
end