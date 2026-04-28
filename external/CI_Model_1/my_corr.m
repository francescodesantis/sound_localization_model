function rho = my_corr(x,y)


x = x - my_mean(x);
y = y - my_mean(y);


rho = sum(x.*y)/sqrt((sum(x.^2)*sum(y.^2)));






function xm = my_mean(x)
xm = sum(x)/length(x);