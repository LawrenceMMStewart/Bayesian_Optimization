
#MG_init0.jl

"""
Return the tvals and xvals of a given Mackey glass equation with x(t)=0 for all negative t, and x(0)=x_0. Where we have
x'(t)=beta*(x(t-delay))/(1+x^n(t-delay))-gamma*x(t)

Arguments
---------
a
    Starting t value
b
    Ending t value
h 
    stepsize
delay
    delay parameter
n
    exponent parameter
gamma
    feedback parameter
beta 
    pade/ fraction parameter
x0
    x(t) at t=0


"""

function MG_init0(a,b,h,delay,n,gamma,beta,x0)
    #Initialise variables
    nexp=(b-a)/h
    tvals=linspace(a,b,nexp) 
    xvals=zeros(length(tvals))
    xvals[1]=x0
    
    #Discretise delay
    p=delay/h

    #Loop and create values
    for i=2:length(xvals)
    try
        x=xvals[Int(i-1-p)]
        dx=beta*x/(1+x^n)-gamma*xvals[i-1]
        xvals[i]=xvals[i-1]+dx*h
        
    catch
        xvals[i]=xvals[i-1]-gamma*xvals[i-1]*h
        end
    end    

    return tvals, xvals
end
    
