#Error vs time for Bayesian and Random Grid Search 


include("XOR_MD.jl")
include("Kernals.jl")
include("gaussian_process.jl")



#Set the random seed:
srand(1234) #Seed for stalzer srand(1234)

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

N=20   

#Curry the sigmoid functions:

function hyper_curry(h)
    return (x->sigmoid(x,h))
end

function hyper_curry_deriv(h)
    return (x->sigmoid_deriv(x,h))
end



Bayesian_Times=zeros(N)
Random_Times=zeros(N)

# Random Learning Rates Examples ========================================


Random_Learning_Rates=uniform(a,b,N,1)
Random_Hyperparameters=uniform(c,d,N,1)
Random_Mat=cat(2,Random_Learning_Rates,Random_Hyperparameters)
Random_MSE=zeros(N)


#Random_Mat conjoins Random_Learning_Rates and Random_Hyperparameters
# Random_Mat is a Nx2 matrix where Random_Mat[1,:] is the first entry
#with LR_1 and hyperparemeter 1.
i=1

node_function=hyper_curry(Random_Mat[i,2])
node_deriv=hyper_curry_deriv(Random_Mat[i,2])
learning_rate=Random_Mat[i,1]

Random_MSE[i]=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]
println("Epoch Complete")




q=0
for i=1:length(Random_Learning_Rates)
    
    tic()

    node_function=hyper_curry(Random_Mat[i,2])
    node_deriv=hyper_curry_deriv(Random_Mat[i,2])
    learning_rate=Random_Mat[i,1]

    Random_MSE[i]=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]
    println("Epoch Complete")
    
    q+=toc()
    Random_Times[i]=q

end

println("Random Learning Rates Training Completed")







#Bayesian Optimization Examples===================================================

#Here are the points we can pick from in the Optimization


#Initialise Layers and params ==========================================

LR_Test=linspace(a,b,50)
HP_Test=linspace(c,d,50)

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


tic()
#Run first train before Bayesian Optimization:
Bayesian_MSE=[Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]]
tq=toc()
Bayesian_Times[1]=tq
#Begin Bayesian Optimization:





#pre-run to remove time bias ==========================================
kr=2
D=[(Bayesian_Points[i],Bayesian_MSE[i]) for i=1:length(Bayesian_Points)]
println("ok")
mu, sigma, D=gaussian_process_chol(std_exp_square_ker,D,1e-6,Test)
# println("Gaussian Process Complete","\r")
mu=reshape(mu,length(mu));
sigma=reshape(sigma,length(sigma))

println("ok")
new_point=findmin(mu-sigma)[2]
println("ok")
#Here we will need to change the number 2 to k 
Bayesian_Points=cat(1,Bayesian_Points,[Test[new_point]])
println("ok")
learning_rate=Bayesian_Points[kr][1]
println("ok")
node_function=hyper_curry(Bayesian_Points[kr][2])
println("ok")
node_deriv=hyper_curry_deriv(Bayesian_Points[kr][2])
println("ok")
value_to_be_appended=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]
println("ok final")
# =====================================================================


#Here we reset the values:
# =====================================================================


#Initialise Layers and params ==========================================

LR_Test=linspace(a,b,50)
HP_Test=linspace(c,d,50)

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


tic()
#Run first train before Bayesian Optimization:
Bayesian_MSE=[Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]]
tq=toc()
Bayesian_Times[1]=tq
 
# =========================================================================





q=0 #preallocate time value at 0
for k=2:N

    tic()
    D=[(Bayesian_Points[i],Bayesian_MSE[i]) for i=1:length(Bayesian_Points)]
    mu, sigma, D=gaussian_process_chol(std_exp_square_ker,D,1e-6,Test)
    # println("Gaussian Process Complete","\r")
    mu=reshape(mu,length(mu));
    sigma=reshape(sigma,length(sigma))

    new_point=findmin(mu-sigma)[2]

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
        q+=toc()
        Bayesian_Times[k]=q
        break
    end

    q+=toc()
    Bayesian_Times[k]=q


end

Bayesian_Times2=Bayesian_Times[1:length(Bayesian_MSE)]


# Bayesian Plotting =========================================================


println(" The optimium is located at ",Bayesian_Points[end])




println("Bayesian Times =",Bayesian_Times2)
println("Random_Times = ", Random_Times)

using PyPlot
# fig = figure("pyplot_subplot_mixed",figsize=(7,7))
# ax=axes()
plot(Bayesian_Times2,Bayesian_MSE,label="Bayesian Optimization")
plot(Random_Times,Random_MSE,label="Random Grid Search")
title("MSE vs Time")
xlabel("Time (s)")
ylabel("MSE")
legend()
grid("off")
show()




