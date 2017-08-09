#Gaussian Process for sin(x+y)
srand(1234)
function uniform(a,b,N,M)
    #Returns a NxM uniformly distributed matrix
    #for random number in range [a,b] we note that it can be generated by: rand() * (b-a) - a 
    #where rand() prints real numbers:
    rand(N,M)*(b-a)+a
end
using Distributions
include("Kernals.jl")
# include("gaussian_process.jl") Reinclude this 

function gaussian_process_chol(ker,D,noise,xrange)
    
    y=map(x->x[2],D)   # these are our y noisy functions
    x=map(x->x[1],D) #These are our x training points
    
    K = cov_gen(ker,x,x)+noise*eye(length(x))

    L=ctranspose(chol(K))
    temp=cov_gen(ker,x,xrange)
    Lk=\(L,cov_gen(ker,x,xrange))

    
    # println("the current size of LK is = ",size(Lk))
    # println("the size of \(L,y) is = ", size(\(L,y) ) )
    # println("size of first column of thingy is ", size(Lk[:,1]))
    # println("the dot product is ", dot(Lk[:,1],\(L,y) ))
    
    mu=[dot(Lk[:,i],(\(L,y))) for i=1:size(Lk)[2] ] #Here we have an error
    
    # println("the length of mu should be 10,000 it is in fact ",size(mu))

    K_=cov_gen(ker,xrange,xrange)+eye(length(xrange))*noise
    element1=diag(K_)
    s2=diag(K_)-[ sum( (Lk.*Lk)[:,i] ) for i=1:size(Lk)[2] ]  
    sigma=sqrt(s2) 
    return (mu,sigma,D)
  
end



Xtest=linspace(-pi,pi,100)
Ytest=linspace(-pi,pi,100)
Test=gen_points([Xtest,Ytest])[1]
sin_test = map(i -> sin(i[1]+i[2]),Test)


#for plotting:

cix=map(x->x[1],Test)
ciy=map(x->x[2],Test)

N=40


Xsample=uniform(-pi,pi,N,1) #if uniform need to add (,1) to specify dimension
Ysample=uniform(-pi,pi,N,1)

Randomsamp = zeros(N,2)

for (i,x) in enumerate(Xsample)
    Randomsamp[i,1] = Xsample[i]
    Randomsamp[i,2] = Ysample[i]
end


#Here Randomsamp is the (x,y) vectors:

xrange=Test #delete thus line


noise_dist=Normal(0,10.0^(-6))
noise=1e-6
# y=map(i-> sin( i[1]+i[2]),Randomsamp)+rand(noise_dist)
Y=[sin(Randomsamp[i,1]+Randomsamp[i,2])+rand(noise_dist) for i=1:size(Randomsamp)[1]]


# K=cov_gen(std_exp_square_ker,Test,Test)+eye(length(Test))*1e-6
 

#Our samples are Randomsamp and y


#print(Randomsamp[1,:]) this gives you the array of each entry of Randomsamp

D=[(Randomsamp[i,:],Y[i]) for i=1:length(Y)];

mu,sigma,D = gaussian_process_chol(std_exp_square_ker,D,1e-6,Test);
mu=reshape(mu,length(mu));
sigma=reshape(sigma,length(sigma));
# an entry from `D is in the form ([x,y],sin(x+y))

# x=map(x->x[1],D)
# print(size(x)) elements of x are in arrays;

y=map(x->x[2],D);   # these are our y noisy function
x=map(x->x[1],D); #These are our x training points (dont forget they come as arrays)
x1=map(p->p[1],x)
x2=map(p->p[2],x)


# println("x = ",x)
# println("y = ",y)
# println("x1= ",x1)
# println("similarly for x2")
# println("The shape of mu and sigma ==", size(mu),size(sigma))


# println("mu= ",mu)
# # println("sigma = ",sigma)
# println("mu and sigma should have thee same shape if this is true",size(mu)==size(sigma))




using PyPlot
fig = figure("pyplot_plot",figsize=(5,5))
ax = axes()

surf(x1,x2,y,alpha=0.8)
surf(cix,ciy,sin_test,alpha=0.5)
surf(cix,ciy,mu+2*sigma,alpha=0.3) #CI intervals:
surf(cix,ciy,mu-2*sigma,alpha=0.3) #CI intervals:


title("Gaussian Process Sin(x+y)") 
# ylabel("f(x)")
# xlabel("x")
grid("off")
show()



















#function stuff

# y=map(x->x[2],D)   # these are our y noisy functions
# x=map(x->x[1],D) #These are our x training points
    
# K = cov_gen(ker,x,x)+noise*eye(length(x))

# L=ctranspose(chol(K))
# temp=cov_gen(ker,x,xrange)
# Lk=\(L,cov_gen(ker,x,xrange))
# mu=[vecdot(transpose(Lk)[i,:],(\(L,y))) for i=1:size(Lk)[1] ]
# K_=cov_gen(ker,xrange,xrange)+eye(length(xrange))*noise

# println("The dimensions of K_ are ", size(K_))
# println("Hence the size of diag K_ should be", size(K_)[1])

# element1=diag(K_)
# println("In fact the actual size is ", size(element1))
# sigma=sqrt(s2) 
# s2=diag(K_)-[ sum( (Lk.*Lk)[:,i] ) for i=1:size(Lk)[2] ]   #There isnt a good sum function to make a  vector where each entry is the sum of a row
# mu,sigma,D = gaussian_process_chol(std_exp_square_ker,D,1e-6,Test);
# mu=reshape(mu,length(mu));
# sigma=reshape(sigma,length(sigma));


# return (mu,sigma,D)
