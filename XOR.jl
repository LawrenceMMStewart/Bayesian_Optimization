# Set random seed:
srand(1234)




#uniform.jl

"""
Create a NxM uniformly distributed matrix using the method rand()*(b-a)-a where a,b are the Unif Parameters.

Arguments
---------
a
    Lower Bound 
b
    Upper Bound
N
    Row Dimension
M 
    Column Dimension

"""
function uniform(a,b,N,M)
    #Returns a NxM uniformly distributed matrix
    #for random number in range [a,b] we note that it can be generated by: rand() * (b-a) - a 
    #where rand() prints real numbers:
    rand(N,M)*(b-a)+a
end






#sigmoid.jl

"""
Return the sigmoid of x:

Arguments
---------
x
    input

"""

#Later on we will add a hyperparemeter to this function:

function sigmoid(x)
    return 1/(1+exp(-x))
    end






#sigmoid_deriv.jl

"""
Return the derivative of the sigmoid of x:

Arguments
---------
x
    input

"""

function sigmoid_deriv(x)
    return 1/(1+exp(-x))*(1-1/(1+exp(-x)))
end






#Train_Neural_Net.jl

"""
Train the XOR neural net for a given number of epochs:

Arguments
---------
epochs
    number of times the network will foward and back propagate
Layer_1
    weighting for hidden layer 
Layer_2
    weighting for output layer 
learning_rate
    Learning Rate for Gradient Descent
"""

function Train_Neural_Net_Loop(epochs,Layer_1,Layer_2,learning_rate)

    #Initialise XOR truth values:
    X=[ 0 0 ; 0 1 ; 1 0 ; 1 1] #When selected X[i,:] we have column vectors so need to transpose for matmult
    Y=[ 0;1;1;0]

    MSE=0

    for i=1:epochs #For the weight matrix wij is the weight for input i going to Neuron j:

        s_1=map(sigmoid,X*Layer_1) #Applies Sigmoid to X*Layer_1 the weighting matrix
        s_2=map(sigmoid,s_1*Layer_2) #Applies Sigmoid to a2 * second weighting matrix
        direct_error=s_2-Y
        MSE=0.5*sum(direct_error.*direct_error) #element wise squares direct_error then summs up and divides by length


    #Begin updating: 
    delta_outer=-1.0*direct_error.*map(sigmoid_deriv,s_1*Layer_2) #this is the delta (which is the raw loss element-wise
                                                        #multiplied by z(3) see explanation in readme 
                                                        
    delta_inner=delta_outer*transpose(Layer_2).*map(sigmoid_deriv,X*Layer_1)  
    # map(sigmoid_deriv,X*Layer_1) is f'(z2)




    #Update the weights:

    Layer_2 +=learning_rate*transpose(s_1)*delta_outer #Outer layer has been updated. 
    Layer_1 += learning_rate*transpose(X)*delta_inner

  
    print(string(MSE,"\r"))

    
    end
    return Layer_1, Layer_2, MSE
    end








#Train_Neural_Net.jl

"""
Train the XOR neural net for a given mean square error it needs to reach:

Arguments
---------
MSE_Min
    Error the network trains till
Layer_1
    weighting for hidden layer 
Layer_2
    weighting for output layer 
learning_rate
    Learning Rate for Gradient Descent
"""

function Train_Neural_Net_MSE(MSE_Min,Layer_1,Layer_2,learning_rate)

    #Initialise XOR truth values:
    X=[ 0 0 ; 0 1 ; 1 0 ; 1 1] #When selected X[i,:] we have column vectors so need to transpose for matmult
    Y=[ 0;1;1;0]

    MSE=MSE_Min+1

    while MSE>=MSE_Min #For the weight matrix wij is the weight for input i going to Neuron j:

        s_1=map(sigmoid,X*Layer_1) #Applies Sigmoid to X*Layer_1 the weighting matrix
        s_2=map(sigmoid,s_1*Layer_2) #Applies Sigmoid to a2 * second weighting matrix
        direct_error=s_2-Y
        MSE=0.5*sum(direct_error.*direct_error) #element wise squares direct_error then summs up and divides by length


    #Begin updating: 
    delta_outer=-1.0*direct_error.*map(sigmoid_deriv,s_1*Layer_2) #this is the delta (which is the raw loss element-wise
                                                        #multiplied by z(3) see explanation in readme 
                                                        
    delta_inner=delta_outer*transpose(Layer_2).*map(sigmoid_deriv,X*Layer_1)  
    # map(sigmoid_deriv,X*Layer_1) is f'(z2)




    #Update the weights:

    Layer_2 +=learning_rate*transpose(s_1)*delta_outer #Outer layer has been updated. 
    Layer_1 += learning_rate*transpose(X)*delta_inner

  
    print(string(MSE,"\r"))

    
    end
    return Layer_1, Layer_2, MSE
    end







#XOR_Net.jl

"""
Output the values of a given XOR_Net

Arguments
---------
M
    Input for the XOR
w1
    Layer 1
w2
    Layer 2

"""

function XOR_Net(M,w1,w2)
    #Where M is a matrix that 
    s_1=map(sigmoid,M*w1) #Applies Sigmoid to X*Layer_1 the weighting matrix
    s_2=map(sigmoid,s_1*w2)
    return s_2

end



#----------------------------- We can add tanh functions and hyperparameters-----------#


# #TEST RUN:---

# #Initialise Layers of Neuron Weights:


Layer_1=uniform(0,1,2,2) #(In video above W1)
Layer_2=uniform(0,1,2,1) #Column vector for the outer layer (in video above W2)

# #Initialise Learning Rate:

# learning_rate1=0.1
# learning_rate2=0.01
# # learning_rate3=0.4










# # w1, w2, MSE1 =Train_Neural_Net_Loop(100000,Layer_1,Layer_2,learning_rate1)
# x1, x2, MSE2 =Train_Neural_Net_Loop(1000000,Layer_1,Layer_2,learning_rate2)
# # z1, z2, MSE3=Train_Neural_Net_Loop(100000,Layer_1,Layer_2,learning_rate3)



# print(string(MSE1,""),string(MSE2,""),string(MSE3,""))




# # #================================================================================================#
# # #If kernal has hyper-parameter: set it here and uncomment the curry function named ker:
# # # function c_ker(hyp,ker)
# # #     return ((x,y) -> ker(x,y,hyp))
# # # end
# # # new_kernal = c_ker(0.1,hyper_exp_square_ker)

