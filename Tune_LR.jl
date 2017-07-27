#We will use Bayesian Optimization to find the optimum learning rate:


include("XOR.jl")
include("Kernals.jl")
include("gaussian_process.jl")


#Set the random seed:
srand(1234) #Seed for stalzer srand(1234)

"""
We will compare the effect of randomly selecting a learning rate vs the use
of Bayesian Optimization for finding the optimial LR on the MSE. Suppose we 
have limited computing time of 100000 epochs and that we have N tries to 
minimise the MSE. Let us say that the learning rate is between a and b

"""



#Initialise Layers and params ==========================================



Layer_1=uniform(0,1,2,2) 

Layer_2=uniform(0,1,2,1) 

epochs=1000   #1000
a=0.0001  #Change to 0.001
b=3.0       #change back to 3
N=25    #change to 25


# Random Learning Rates Examples ========================================

Random_Learning_Rates=uniform(a,b,N,1)
Random_MSE=zeros(N)

for i=1:length(Random_Learning_Rates)
    Random_MSE[i]=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,Random_Learning_Rates[i])[3]
    println("Epoch Complete")
end

println("Random_Learning_Rates Training Complete")





# Bayesian Optimization Example ========================================


LR_Test=linspace(a,b,1000) #These are the learning rates we can choose



#We first have to pick a random point to begin the Optimization:
Bayesian_Learning_Rates=uniform(a,b,1,1)

#Update the Bayesian_MSE:
Bayesian_MSE=[Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,Bayesian_Learning_Rates[1])[3]]



#All good up to here, delete me:

mu=zeros(size(LR_Test))
sigma=zeros(size(LR_Test))
for k=2:N


#     #Use Bayesian OPtimization to find the next best rate to choose & evaluate
    D=[(Bayesian_Learning_Rates[i],Bayesian_MSE[i]) for i=1:length(Bayesian_Learning_Rates)]
    mu,sigma,D = gaussian_process(std_exp_square_ker,D,1e-6,LR_Test)
    mu=reshape(mu,length(mu));
    sigma=reshape(sigma,length(sigma));

    #     #We take the LCB /SDO for our utility function:
    #     #for now we will consider plus/minus one standard deviation:

    new_learning_rate_position=findmin(mu-sigma)[2] #Second arguement of finmin is position


    Bayesian_Learning_Rates=cat(1,Bayesian_Learning_Rates,[LR_Test[new_learning_rate_position]])
    value_to_be_appeneded=Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,Bayesian_Learning_Rates[k])[3]
    if value_to_be_appeneded != Bayesian_MSE[k-1]
        Bayesian_MSE=cat(1,Bayesian_MSE,[value_to_be_appeneded])   
        println("Epoch Complete")
    else
        println("Found Optimum")
        Bayesian_Learning_Rates=Bayesian_Learning_Rates[1:length(Bayesian_Learning_Rates)-1]
        break
    end

end


println("Bayesian_Learning_Rates Training Complete")

print(minimum(Bayesian_MSE))
print(minimum(Random_MSE))


#gaussian process plotting
using PyPlot
fig = figure("pyplot_plot",figsize=(5,5))
ax = axes()
fill_between(LR_Test,mu-sigma,mu+sigma,facecolor="#a6a6a6")#This fills confindence interval for two standard 
#deviations, currently the variance is one (as we take off diagonal)
plot(Bayesian_Learning_Rates,Bayesian_MSE,linewidth=0,marker="o")

title("Gaussian Process for Learning Rates") 
ylabel("f(x)")
xlabel("x")
grid("on")
show()






# #Plotting MSE vs Learning Rates


using PyPlot
fig = figure("pyplot_subplot_mixed",figsize=(7,7))
ax=axes()

plot(Bayesian_Learning_Rates, Bayesian_MSE,linewidth=0,marker="o")
# for i=1:length(Bayesian_Learning_Rates)
#     annotate(string(i), xy=[Bayesian_Learning_Rates[i];Bayesian_MSE[i]])  #Annotate from here
# end
plot(Random_Learning_Rates, Random_MSE,linewidth=0,marker="o")

title("Final MSE of 10000 Epoch ANN with Varied Learning Rates")
ylabel("MSE")
xlabel("learning rate")
grid("on")
show()

