%% Cart Pole Dynamics (Lagrange Eqn.) with Animation
h = .01;

figure(1)
cart = polyshape([0 0 2 2 ],[2 0 0 2]);
pole = polyshape([0 0 .5 .5 ],[8 2 2 8]);

% Parameters 
mp = 1; 
m = 5 ; 
g = -9.81; 
L = 1;
F_x = 1 ;

x = 0; 
x_dot =0; 
x_ddot =0; 

theta =0 ; 
theta_dot =0; 
theta_ddot =0; 
 
for k=1:1:150

    
    x_num = (mp * g * sin(theta) * cos(theta) - mp * L * sin(theta) * (theta_dot) ^ 2 + F_x);
    x_den = ((m + mp) - mp * (cos(theta)) ^ 2);
    x_ddot = (x_num / x_den);
    
    theta_num = (mp * L * sin(theta) * cos(theta) * (theta_dot) ^ 2 - cos(theta) * F_x - (m + mp) * g * sin(theta));
    theta_den = (mp * L * (cos(theta)) ^ 2 - (m + mp) * L);
    theta_ddot = (theta_num / theta_den);
    
    
    x_dot = x_dot + h*x_ddot; 
    x = x + h*x_dot; 
    
    theta_dot = theta_dot + h*theta_ddot; 
    theta = theta + h *theta_dot;
    
    
    cart = translate(cart,[x,0]); 
    pole = rotate(pole,theta);
    
    plot(cart)
%     plot(pole)
    
    xlim([0 10])
    ylim([0 10])
    pause(.2);
%     drawnow limitrate
end 
