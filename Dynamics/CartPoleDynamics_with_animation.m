%% Cart Pole Dynamics (Lagrange Eqn.) with Animation
h = .01;

figure(1)
cart = polyshape([0 0 2 2 ],[2 0 0 2]);
pole = polyshape([1 1 1.5 1.5 ],[8 2 2 8]);


% Parameters 
mp = 1; 
m = 5 ; 
g = 9.81; 
L = 1;
% F_x = 1 ;

x = 0;
x_dot =0; 
x_ddot =0; 

theta =0 ; 
theta_dot =5; 
theta_ddot =0; 

h=.1;
k_list = [1:h:300];
f_input = 5*sin(.5*k_list);
 
for k=1:1:100
    
    F_x = f_input(k);
    
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
    pole = rotate(pole,theta, [1.25, 2]);
    
    figure(1)
    
    
%     plot(cart)
    hold on
    ps =  plot(pole);
    ps.FaceColor = '#808080';
    cs = plot(cart); 
    cs.FaceColor = '#8B4513';
    
    hold off
   
    xlim([0 10])
    ylim([0 10])
    pause(.2);
%     drawnow limitrate
end 
