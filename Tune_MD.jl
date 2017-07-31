#Here we will optimize the learning rate and sigmoid parameter for the XOR neural network

include("XOR_MD.jl")
include("Kernals.jl")
include("gaussian_process.jl")



#Set the random seed:
srand(1234) #Seed for stalzer srand(1234)

"""
We will compare the effect of randomly selecting a learning rate and sigmoid 
hyperparemeter vs the use of Bayesian Optimization for finding the optimial LR on the MSE. 
Suppose we have limited computing time of 100000 epochs and that we have N tries to 
minimise the MSE. Let us say that the learning rate is between a and b

"""

#Initialise Layers and params ==========================================

Layer_1=uniform(0,1,2,2) 

Layer_2=uniform(0,1,2,1) 

epochs=10000   #1000
a=0.0001  #Change to 0.001
b=3.0

c=0.001
d=1

N=30   #change to 25

#Preallocate the curried functions:

function hyper_curry(h)
    return (x->sigmoid(x,h))
end

function hyper_curry_deriv(h)
    return (x->sigmoid_deriv(x,h))
end



# # Random Learning Rates Examples ========================================



# Random_Learning_Rates=uniform(a,b,N,1)
# Random_Hyperparameters=uniform(c,d,N,1)
# Random_Mat=cat(2,Random_Learning_Rates,Random_Hyperparameters)
# Random_MSE=zeros(N)


# #Random_Mat conjoins Random_Learning_Rates and Random_Hyperparameters
# # Random_Mat is a Nx2 matrix where Random_Mat[1,:] is the first entry
# #with LR_1 and hyperparemeter 1.



# for i=1:length(Random_Learning_Rates)
#     node_function=hyper_curry(Random_Mat[i,2])
#     node_deriv=hyper_curry_deriv(Random_Mat[i,2])
#     learning_rate=Random_Mat[i,1]
#     Random_MSE[i]=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]
#     println("Epoch Complete")
# end

# println("Random Learning Rates Training Completed")
# print("The Minimum MSE that was found Randomly = ", minimum(Random_MSE))




# # Random Test Plotting=========================================================

# using PyPlot
# # fig = figure("pyplot_subplot_mixed",figsize=(7,7))
# ax=axes()

# surf(reshape(Random_Learning_Rates,size(Random_MSE)),reshape(Random_Hyperparameters,size(Random_MSE)),Random_MSE,alpha=0.7)
# title("MSE Plot for varied Sigmoid Hyper-Parameters and Learning Rates")
# xlabel("Learning Rates")
# ylabel("Hyper-Parameters")
# zlabel("Mean Square Error")
# grid("on")
# show()




#Bayesian Optimization Examples===================================================

#Here are the points we can pick from in the Optimization

LR_Test=linspace(a,b,1000)
HP_Test=linspace(c,d,1000)

#Here is the carteisan product of these written as a vector
Test=gen_points([LR_Test,HP_Test])[1]


# K=cov_gen(std_exp_square_ker,Test,Test)+eye(length(Test))*1e-6 #probably wont need this:


#We first have to pick a random point to begin bayesian optimization:

#currently starts with the midpoint, possibly randomise this:
Bayesian_Points=[Test[Int(round(length(Test)/2))]]



#Bayesian_Points is an vector of arrays where in each array first entry is LR second entry is Hyper-Parameters:


#Define hyperparemeter function:
node_function=hyper_curry(Bayesian_Points[1][2])
node_deriv=hyper_curry_deriv(Bayesian_Points[1][2])
learning_rate=Bayesian_Points[1][1]


Bayesian_MSE=[Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]]






