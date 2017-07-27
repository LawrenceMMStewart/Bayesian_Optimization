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

function catty(a,b)
    return cat(1,a,b)
end

#comb.jl

"""
Given two arrays with single arrays inside generate the cartesian product set
Coded by: Vandan Parmar
Arguments
---------

a
    array of array 1
b
    array of array 2

"""

function comb(a,b)


    function op1(a,b)
        return [a,b]
    end

    function op2(a,b)
        return cat(1,[a],b)
    end

    function op3(a,b)
        return cat(1,a,[b]) 
    end

    function op4(a,b)
        return cat(1,a,b)
    end
    a=a[1]


    #shout out to nando
    b=b[1] #Set the array of arrays to be an array so Vandan does not commit at this current time
 
    if (size(a[1])==())
        if(size(b[1])==())
            op_i = op1
        else
   
            op_i = op2
        end
    else
        if(size(b[1])==())
            op_i = op3
        else
            op_i = op4
        end
    end

    # print(cat(,map(y -> op_i(1,y),b)))
    print("\n")
    print("\n")
    toReturn = []
    map(x -> map(y -> push!(toReturn,op_i(x,y)),b), a)
    # show(string(toReturn))

    # toReturn = []

    # for ai in a
    #     for bj in b
    #         push!(toReturn,op_i(ai,bj))
    #         print(ai,"\r")
    #     end
    # end

    return [toReturn]
end





#gen_points.jl

"""
Given an array of arrays where each array is the set of values each variable can take, for example 
the first array may be learning rate (size 10), the second may be Hyper-parameter 1 size(1000), ect ect,
the function gen_points will generate the set of all possible points considering a point in dimenstion R^n
where n is the number of variables one uses.

Arguments
---------
S
    Array of all arrays containing variable values

"""

function gen_points(S)
    print("gen_points")
    if size(S)[1]==1
        return S[1]
    else
        divider=convert(Int64,round( size(S)[1] /2) )    #Rounds up number of sets divided by two
        
        ar1=S[1:divider]
        ar2=S[divider+1:end] #Splits S into two
        
        if size(ar2)[1]==1
            if size(ar1)[1]==1
                return comb(ar1,ar2)
            else 
                ar1=gen_points(ar1)
                return comb(ar1,ar2)
            end
        else
            ar1=gen_points(ar1)
            ar2=gen_points(ar2)
            return comb(ar1,ar2)
            
        end
    end
end





