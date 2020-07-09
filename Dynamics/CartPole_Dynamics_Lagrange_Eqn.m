%% Cart-Pole Dynamics via Euler-Lagrange Equations


% Initial Conditions
x_0 = 0;
x_dot_0 = 0.0;

theta_0 = 0.001;
theta_dot_0 = -2;

input_0 = 0;

% Empty Variables
x = [x_0];
x_dot = [x_dot_0];
x_ddot = [];

theta = [theta_0];
theta_dot = [theta_dot_0];
theta_ddot = [];

input = input_0;
h = .001;


states = [x_0, x_dot_0, theta_0, theta_dot_0];
[x_ddot(1), theta_ddot(1)] = accel_model(states, input);

sim_len = 360;

for k =1:1:sim_len    
   
   
    x_dot(k+1) = x_dot(k)+h* x_ddot(k);
    theta_dot(k+1) = theta_dot(k)+h*theta_ddot(k);
   
    x(k+1) = x(k) + h*x_dot(k);
    theta(k+1) = theta(k) + h*theta_dot(k);

    states = [x(k+1), x_dot(k+1), theta(k+1), theta_dot(k+1)];  
   
    if k == sim_len
        break
    end

   
   
    [x_ddot(k+2), theta_ddot(k+2)] = accel_model(states, input);    
end


   
% Plots
t = [1:1:(sim_len+1)];

figure(1)
subplot(3,1,1)
plot(t,x)
title("Linear Position vs Time")
xlabel("Time")
ylabel("linear Position")

subplot(3,1,2)
plot(t,x_dot)
title("Linear Velocity vs Time")
xlabel("Time")
ylabel("linear Velocity")

subplot(3,1,3)
plot(t,x_ddot)
title("Linear Acceleration vs Time")
xlabel("Time")
ylabel("linear Acceleration")


figure(2)
subplot(3,1,1)
plot(t,theta)
title("Rotational Position vs Time")
xlabel("Time")
ylabel("Rotational Position")

subplot(3,1,2)
plot(t,theta_dot)
title("Rotational Velocity vs Time")
xlabel("Time")
ylabel("Rotational Velocity")



subplot(3,1,3)
plot(t,theta_ddot)
title("Rotational Acceleration vs Time")
xlabel("Time")
ylabel("Rotational Acceleration")




%% Equations of Motion


function [linear_acceleration, rotational_acceleration] = accel_model(states, input)

% Parameters
mp = 1;
m = 10 ;
g = -9.81;
L = .5;

% States = [x, x_dot, theta, theta_dot];
theta = states(3);
theta_dot = states(4);
F_x = input;

x_num = (mp*g*sin(theta)*cos(theta)-mp*L*sin(theta)*(theta_dot)^2 + F_x);
x_den = ((m + mp)-mp*(cos(theta))^2);
x_ddot = (x_num/x_den);

theta_num =(mp*L*sin(theta)*cos(theta)*(theta_dot)^2 - cos(theta)*F_x -(m+mp)*g*sin(theta));
theta_den = (mp*L*(cos(theta))^2-(m+mp)*L);
theta_ddot = (theta_num/theta_den);

rotational_acceleration = theta_ddot;
linear_acceleration = x_ddot;
end 
