function matern_ker(x,y,h)
    if size(x)!==size(y)
        error("Error: Input vectors must have the same dimension and shape")
    end

    return ((0.5^(h-1))/(gamma(h)))*(2*sqrt(h)*sqrt(dot(x-y,x-y)))^h*besselj(h,2*sqrt(h)*sqrt(dot(x-y,x-y)))

end
print(matern_ker([1,1,1],[0,0,0],3))