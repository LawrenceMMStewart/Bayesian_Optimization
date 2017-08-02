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

epochs=1000   #1000
a=0.001  #Change to 0.001
b=1

c=0.001
d=1

N=20   #change to 25

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





# Random Test Plotting=========================================================

using PyPlot
# fig = figure("pyplot_subplot_mixed",figsize=(7,7))
ax=axes()

surf(reshape(Random_Learning_Rates,size(Random_MSE)),reshape(Random_Hyperparameters,size(Random_MSE)),Random_MSE,alpha=0.7)
title("MSE Plot for varied Sigmoid Hyper-Parameters and Learning Rates")
xlabel("Learning Rates")
ylabel("Hyper-Parameters")
zlabel("Mean Square Error")
grid("on")
show()




#Bayesian Optimization Examples===================================================

#Here are the points we can pick from in the Optimization

LR_Test=linspace(a,b,100)
HP_Test=linspace(c,d,100)

#Here is the carteisan product of these written as a vector
Test=gen_points([LR_Test,HP_Test])[1]





#We first have to pick a random point to begin bayesian optimization:

#currently starts with the midpoint, possibly randomise this:
Bayesian_Points=[Test[Int(round(length(Test)/2))]]



#Bayesian_Points is an vector of arrays where in each array first entry is LR second entry is Hyper-Parameters:


#Define hyperparemeter functions:
node_function=hyper_curry(Bayesian_Points[1][2])
node_deriv=hyper_curry_deriv(Bayesian_Points[1][2])

#Define Learning Rate:
learning_rate=Bayesian_Points[1][1]


#Run first train before Bayesian Optimization:
Bayesian_MSE=[Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]]

#Begin Bayesian Optimization:

for k=2:N
    D=[(Bayesian_Points[i],Bayesian_MSE[i]) for i=1:length(Bayesian_Points)]
    mu, sigma, D=gaussian_process(std_exp_square_ker,D,1e-6,Test)
    println("Gaussian Process Complete","\r")
    mu=reshape(mu,length(mu));
    sigma=reshape(sigma,length(sigma));

    new_point=findmin(mu-0.25*sigma)[2]


    #Here we will need to change the number 2 to k 
    Bayesian_Points=cat(1,Bayesian_Points,[Test[new_point]])

    learning_rate=Bayesian_Points[k][1]
 
    node_function=hyper_curry(Bayesian_Points[k][2])

    node_deriv=hyper_curry_deriv(Bayesian_Points[k][2])

    value_to_be_appended=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]

    

    if value_to_be_appended !=Bayesian_MSE[k-1]
        Bayesian_MSE=cat(1,Bayesian_MSE,[value_to_be_appended])
        println("Epoch Complete")
    else
        println("Found Optimum on the ", k-1, " iteration of ", N, " iterations")
        Bayesian_Points=Bayesian_Points[1:length(Bayesian_Points)-1]
        break
    end

end



# Bayesian Plotting =========================================================




LR=[Bayesian_Points[i][1] for i=1:length(Bayesian_Points)]
HP=[Bayesian_Points[i][2] for i=1:length(Bayesian_Points)]




#Move this to the bottom
println("Bayesian_Learning_Rates Training Complete")
println("The minimum MSE by Bayesian Optimization was", minimum(Bayesian_MSE))
println("The minnmum MSE by Random Selection was", minimum(Random_MSE))

using PyPlot
# fig = figure("pyplot_subplot_mixed",figsize=(7,7))
# ax=axes()
surf(LR,HP,Bayesian_MSE,alpha=0.7)

title("MSE Plot for Optimized Sigmoid Hyper-Parameters and Learning Rates")
xlabel("Learning Rates")
ylabel("Hyper-Parameters")
zlabel("Mean Square Error")

grid("off")
show()

"""
============================================== Results ==========================================



_________ Experiment 1_____________________

#We let the learning rate and hyper-parameter be between 0.001 and 1
#we sample 25 times with 1000 epochs:

consider µ+a*sigma where a is in (0,1). This is for standard LCB bayesian optimization:

==================================================================================
a=1 -----> 

Found Optimum on the 6 iteration of 20 iterations
The minimum MSE by Bayesian Optimization was0.1766613895128894
The minnmum MSE by Random Selection was0.2931509389546021
 
==================================================================================
a=0.75 ---->

Found Optimum on the 6 iteration of 20 iterations

The minimum MSE by Bayesian Optimization was0.1766613895128894
The minnmum MSE by Random Selection was0.2931509389546021
==================================================================================
a=0.5 -->

Found Optimum on the 4 iteration of 20 iterations

The minimum MSE by Bayesian Optimization was0.1766613895128894
The minnmum MSE by Random Selection was0.2931509389546021
==================================================================================
a=0.25 -->

Found Optimum on the 4 iteration of 20 iterations 

The minimum MSE by Bayesian Optimization was0.1766613895128894
The minnmum MSE by Random Selection was0.2931509389546021
==================================================================================

Shows the exploitation of the network (µ(x)) vs the exploration (sig(x)) does not 
really matter in terms of the final MSE however the smaller a found it in less.
"""
