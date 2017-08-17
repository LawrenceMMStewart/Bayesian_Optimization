include("Kernals.jl")
include("gaussian_process.jl")


function uniform(a,b,N)
    rand(N)*(b-a)+a
end

using Distributions
n=50 #number of test points
N=10;#Number of training points
Xtest=linspace(-5,5,n); #Xtest are all of available points to check on the axis
X=uniform(-5,5,N); #X is the values we will sample at and train
noise_dist=Normal(0,10.0^(-6)) #Distribution of Noise
Y=sin(X)+rand(noise_dist) #Y is our function values with noise 
K=cov_gen(std_exp_square_ker,Xtest,Xtest)+eye(length(Xtest))*1e-6
L=ctranspose(chol(K));
dist=MvNormal(zeros(n),eye(K)) # We draw f from N(0,I)L which is the same as N(0,K)
fprior=L*rand(dist,3); #Each column is a function



using PyPlot
fig = figure("pyplot_plot",figsize=(5,5))
ax = axes()
fill_between(Xtest,-2*diag(K),2*diag(K),facecolor="#a6a6a6")#This fills confindence interval for two standard 
#deviations, currently the variance is one (as we take off diagonal)
plot(Xtest,fprior,alpha=0.75)
title("N=3 Multivariate Gaussians")
ylabel("f(x)")
xlabel("x")
grid("off")
show()



D=[(X[i],Y[i]) for i=1:length(X)];
mu,sigma,D = gaussian_process(std_exp_square_ker,D,1e-6,Xtest);
mu=reshape(mu,length(mu));
sigma=reshape(sigma,length(sigma));
y=map(x->x[2],D);   # these are our y noisy function
x=map(x->x[1],D); #These are our x training points
using PyPlot
fig = figure("pyplot_plot",figsize=(5,5))
ax = axes()
fill_between(Xtest,mu-2*sigma,mu+2*sigma,facecolor="#a6a6a6")#This fills confindence interval for two standard 
#deviations, currently the variance is one (as we take off diagonal)
plot(x,y,linewidth=0,marker="o")

title("N=3 Multivariate Gaussians") 
ylabel("f(x)")
xlabel("x")
grid("off")
show()
