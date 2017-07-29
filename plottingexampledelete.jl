using PyPlot

####################
##  Prepare Data  ##
####################
u = linspace(0.0,2pi,300);
v = linspace(0.0,pi,300);

lu = length(u);
lv = length(v);

x = zeros(lu,lv);
y = zeros(lu,lv);
z = zeros(lu,lv);

for uu=1:lu
	for vv=1:lv
		x[uu,vv]= cos(u[uu])*sin(v[vv]);
		y[uu,vv]= sin(u[uu])*sin(v[vv]);
		z[uu,vv]= cos(v[vv]);
	end
end

#######################
##  Generate Colors  ##
#######################
colors = rand(lu,lv,3)

############
##  Plot  ##
############
surf(x,y,z,facecolors=colors);
show()