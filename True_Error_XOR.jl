#This file samples far more random points in case the viewer wants to see a more complete picture of the networks error function:
#Not very useful but sometimes it is good to know:


#True_error_XOR


include("XOR_MD.jl")
include("Kernals.jl")
include("gaussian_process.jl")



#Set the random seed:
srand(1234) 

"""
We will compare the effect of randomly selecting a learning rate and sigmoid 
hyperparameter vs the use of Bayesian Optimization for finding the optimial values wrt the MSE. 
Suppose we have limited computing time of 100000 epochs and that we have N tries to 
minimise the MSE. Let us say that the learning rate is between a and b

"""

#Initialise Layers and params ==========================================

Layer_1=uniform(0,1,2,2) 

Layer_2=uniform(0,1,2,1) 

epochs=1000   # was 1000 below, please change back
a=0.001  #Change to 0.001
b=1

c=0.001 #Was the same as above
d=1

N=1000 

#Curry the sigmoid functions:

function hyper_curry(h)
    return (x->sigmoid(x,h))
end

function hyper_curry_deriv(h)
    return (x->sigmoid_deriv(x,h))
end



# Random Learning Rates Examples ========================================


    Random_Learning_Rates=uniform(a,b,N,1)
    Random_Hyperparameters=uniform(c,d,N,1)
    Random_Mat=cat(2,Random_Learning_Rates,Random_Hyperparameters)
    Random_MSE=zeros(N)


    #Random_Mat conjoins Random_Learning_Rates and Random_Hyperparameters
    # Random_Mat is a Nx2 matrix where Random_Mat[1,:] is the first entry
    #with LR_1 and hyperparemeter 1.



    for i=1:length(Random_Learning_Rates)
        node_function=hyper_curry(Random_Mat[i,2])
        node_deriv=hyper_curry_deriv(Random_Mat[i,2])
        learning_rate=Random_Mat[i,1]
        Random_MSE[i]=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]
        println("Epoch Complete")
    end

    println("Random Learning Rates Training Completed")
   


using PyPlot
# fig = figure("pyplot_subplot_mixed",figsize=(7,7))
ax=axes()

surf(reshape(Random_Learning_Rates,size(Random_MSE)),reshape(Random_Hyperparameters,size(Random_MSE)),Random_MSE,alpha=0.65,color="#40d5bb")
title("Complete MSE Plot")
xlabel("Learning Rate")
ylabel("Hyper-Parameter")
zlabel("Mean Square Error")
grid("on")
show()
#12a3b4 aqua
println("Minimum MSE Achievable is ",minimum(Random_MSE))
