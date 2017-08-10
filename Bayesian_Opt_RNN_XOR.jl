#Bayesian_Opt_RNN_XOR.jl

#Set the random seed:

srand(1234)

include("Reccurent_XOR.jl")
include("Kernals.jl")
include("gaussian_process.jl")

# Here we will create the reccurent neural network:

Layer_1=uniform(0,1,1,2) 
Layer_2=uniform(0,1,2,1)
recurrent_layer=uniform(0,1,2,2)

Seq_Len=100
epochs=75 #100 was good before


function hyper_curry(h)
    return (x->sigmoid(x,h))
end

function hyper_curry_deriv(h)
    return (x->sigmoid_deriv(x,h))
end





#initialise learning rates and hyper-parameters

a=0.001  #Change to 0.001
b=1
c=0.001
d=1
N=15 #10 does well    

#15 gives Minimum Average Square Error for Random Selection = 0.11429806093691719
#Minimum Average Square Error for Bayesian Optimization = 0.1226313941360362



Random_Learning_Rates=uniform(a,b,N,1)

Random_Hyperparameters=uniform(c,d,N,1)

Random_Mat=cat(2,Random_Learning_Rates,Random_Hyperparameters)
Random_Average_of_Temporal_Square_Errors=zeros(N)



for i=1:length(Random_Learning_Rates)
    node_function=hyper_curry(Random_Mat[i,2])
    node_deriv=hyper_curry_deriv(Random_Mat[i,2])
    learning_rate=Random_Mat[i,1]

    #Here P is the first epoch of the temporal and the last epoch of the temporal SE Values in a 1x2 matrix
    P=Train_Reccurent_Net_Loop(epochs,Layer_1,Layer_2,recurrent_layer,learning_rate,node_function,node_deriv,Seq_Len)

    Random_Average_of_Temporal_Square_Errors[i]=(1/length(P[2]))*sum(P[2])

    #Random_MSE[i]=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate,node_function,node_deriv)[3]
    println("Epoch Complete")
end

println("Random Learning Rates Training Completed")





# Plot the random hyper-parameter and Avergae SE values:


using PyPlot
# fig = figure("pyplot_subplot_mixed",figsize=(7,7))
ax=axes()

surf(reshape(Random_Learning_Rates,size(Random_Average_of_Temporal_Square_Errors)),reshape(Random_Hyperparameters,size(Random_Average_of_Temporal_Square_Errors)),Random_Average_of_Temporal_Square_Errors,alpha=0.7)
title("Average Temporal SE Plot for varied Sigmoid Hyper-Parameters and Learning Rates")
xlabel("Learning Rates")
ylabel("Hyper-Parameters")
zlabel("Mean Square Error")
grid("on")
show()





#Bayesian Optimization Example===================================================

#Here are the points we can pick from in the Optimization
number_of_points=125 # at 75 was 0.12035 vs 0.11

LR_Test=linspace(a,b,number_of_points)
HP_Test=linspace(c,d,number_of_points)

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
P=Train_Reccurent_Net_Loop(epochs,Layer_1,Layer_2,recurrent_layer,learning_rate,node_function,node_deriv,Seq_Len)
Bayesian_Temporal_Means=[(1/length(P[2]))*sum(P[2])]



mu=zeros(number_of_points)
sigma=zeros(number_of_points)
#Begin Bayesian Optimization:

sigma_control=linspace(1,0.1,N) #Here we have added a parameter that fades away with time

@time begin 
    for k=2:N
        D=[(Bayesian_Points[i],Bayesian_Temporal_Means[i]) for i=1:length(Bayesian_Points)]
        mu, sigma, D=gaussian_process_chol(std_exp_square_ker,D,1e-6,Test)  #This line here has been changed to chol version of gaussian_process
        println("Gaussian Process", k, " Complete","\r")
        mu=reshape(mu,length(mu));
        sigma=reshape(sigma,length(sigma))


        new_point=findmin(mu-sigma_control[k]*sigma)[2]

        #Here we will need to change the number 2 to k 
        Bayesian_Points=cat(1,Bayesian_Points,[Test[new_point]])

        learning_rate=Bayesian_Points[k][1]
     
        node_function=hyper_curry(Bayesian_Points[k][2])

        node_deriv=hyper_curry_deriv(Bayesian_Points[k][2])

        P=Train_Reccurent_Net_Loop(epochs,Layer_1,Layer_2,recurrent_layer,learning_rate,node_function,node_deriv,Seq_Len)
        value_to_be_appended=(1/length(P[2]))*sum(P[2])

        

        if value_to_be_appended !=Bayesian_Temporal_Means[k-1]
            Bayesian_Temporal_Means=cat(1,Bayesian_Temporal_Means,[value_to_be_appended])
            println("Epoch ", k, " Complete")
        else
            println("Found Optimum on the ", k-1, " iteration of ", N, " iterations")
            Bayesian_Points=Bayesian_Points[1:length(Bayesian_Points)-1]
            break
        end

    end

    println("Bayesian_Learning_Rates Training Complete")


# Bayesian Plotting =========================================================




LR=[Bayesian_Points[i][1] for i=1:length(Bayesian_Points)]
HP=[Bayesian_Points[i][2] for i=1:length(Bayesian_Points)]




#Move this to the bottom
println("Minimum Average Square Error for Random Selection = ", minimum(Random_Average_of_Temporal_Square_Errors))
println("Maximum Average Square Error for Random Selection = ", maximum(Random_Average_of_Temporal_Square_Errors))
println("Minimum Average Square Error for Bayesian Optimization = ", minimum(Bayesian_Temporal_Means))
println("Maximum Average Square Error for Bayesian Optimization = ", maximum(Bayesian_Temporal_Means))


end


using PyPlot
# fig = figure("pyplot_subplot_mixed",figsize=(7,7))
# ax=axes()
scatter(LR,HP,Bayesian_Temporal_Means)
surf(LR,HP,Bayesian_Temporal_Means,alpha=0.7)
surf(LR,HP,mu+sigma,alpha=0.3) #If it decreases we need to change this to a matrix comprehension
surf(LR,HP,mu-sigma,alpha=0.3)

title("Average Temporal SE Plot for Optimized Sigmoid Hyper-Parameters and Learning Rates")
xlabel("Learning Rates")
ylabel("Hyper-Parameters")
zlabel("Average Square Error")

grid("off")
show()









