include("Kernals.jl")
include("gaussian_process.jl")
include("MG_init0.jl")



#Initialise parameters 
h=0.01;
a=0;
b=120;
delay=17; 
x0=1.2;
n=10;
beta=0.2;
gamma=0.1;


tvals,xvals=MG_init0(a,b,h,delay,n,gamma,beta,x0)        
     
# using PyPlot
# fig = figure("pyplot_plot",figsize=(10,10))
# ax = axes()
# plot(tvals,xvals,alpha=0.75)

# title("Mackey-Glass")
# ylabel("x(t)")
# ylabel("t")
# grid("on")
# show()


# #Here we will take some points of tvals and xvals to be our sample D for the gaussian_process:
# #Here we take every 12000 entries, i.e 100 evenly spaced entrys.
t_train=tvals[1:120:end];
x_train=xvals[1:120:end];


#Choose kernal here:
#================================================================================================#
#If Kernal has no hyper-parameter:
ker=std_exp_square_ker



#================================================================================================#
#If kernal has hyper-parameter: set it here and uncomment the curry function named ker:
#c_ker=std_square_exp_ker
# hyp=10
# function ker(x,y)
#     return c_ker(x,y,hyp)
# end

D=[(t_train[i],x_train[i]) for i=1:length(t_train)];
mu,sigma,D = gaussian_process(ker,D,1e-6,tvals);
mu=reshape(mu,length(mu));
sigma=reshape(sigma,length(sigma));
print(mu[1])

using PyPlot
fig = figure("pyplot_plot",figsize=(5,5))
ax = axes()
fill_between(tvals,mu-2*sigma,mu+2*sigma,facecolor="#a6a6a6")
plot(t_train,x_train,linewidth=0,marker="o")

title("Mackey-Glass Sample Points with CI") 
ylabel("x(t)")
xlabel("t")
grid("off")
show()
print(mu[end])





