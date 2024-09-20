clear all;
rng('default');
filepath = fileparts(mfilename('fullpath'));
parts = strsplit(filepath, filesep);
parent_path = strjoin(parts(1:end-1), filesep);
result_dir = fullfile(parent_path, 'datasets', 'eg3_TwoLinkArm', '001');
rng(0);
copyfile(strcat(mfilename('fullpath'),'.m'),result_dir)

%% Define global parameters
global m_link1 m_motor1 I_link1 I_motor1 m_link2 m_motor2 I_link2 ...
    I_motor2 l1 l2 a1 a2 kr1 kr2 g Fv1 Fv2 Fc1 Fc2 s1 s2 qd Kp Kd ...
    am1 am2 freq1 freq2 th1 th2;

m_link1 = 20; m_motor1 = 2; I_link1 = 5; I_motor1 = 0.01; 
m_link2 = 20; m_motor2 = 2; I_link2 = 5; I_motor2 = 0.01; 
l1 = 0.4; l2 = 0.4; a1 = 0.8; a2 = 0.8; kr1 = 100; kr2 = 100; 
g = 9.81;Fv1 = 40; Fv2 = 40; Fc1 = 2; Fc2 = 2; s1 = 10; s2 = 10;
Kp = 1*eye(2); Kd = 2*eye(2);

num_sine = 1;
predefined_freq = [1];
am1 = 200*ones(1,num_sine);
am2 = 200*ones(1,num_sine);
freq1 = zeros(num_sine,1);
freq2 = zeros(num_sine,1);
freq1(1:size(predefined_freq,2)) = predefined_freq;
freq2(1:size(predefined_freq,2)) = predefined_freq;
if num_sine > size(predefined_freq,1)
    freq1(size(predefined_freq,1)+1:end) = rand(num_sine-size(predefined_freq,1), 1)*10;
    freq2(size(predefined_freq,1)+1:end) = rand(num_sine-size(predefined_freq,1), 1)*10;
end

%% Gaussian noise variances
var_q = 3e-5;
var_q_dot = 3e-5;
var_x = [var_q,var_q,var_q_dot,var_q_dot];

%% Simulation settings
n_state = 4;
n_control = 2;
t_end = 3.2;
sampling_freq = 100;
tspan = linspace(0, t_end, sampling_freq*t_end+1);
theta1_theta2_space = [-pi/6*5, pi/6*5; 
                        -pi/6*5, pi/6*5];
points_per_dim = [20, 20]; % Important!!!
x1 = linspace(theta1_theta2_space(1,1), theta1_theta2_space(1,2),...
    points_per_dim(1));
x2 = linspace(theta1_theta2_space(2,1), theta1_theta2_space(2,2),...
    points_per_dim(2));
[X1,X2] = ndgrid(x1,x2);
initial_states = zeros(prod(points_per_dim),n_state);
initial_states(:,1) = reshape(X1,[],1);
initial_states(:,2) = reshape(X2,[],1);

%% Collect all sinusoidal phases
num_traj = size(initial_states,1);
all_th1 = zeros(num_traj, num_sine);
all_th2 = zeros(num_traj, num_sine);

%% Simulation
X = zeros(size(initial_states,1)*sampling_freq*t_end,n_state); % noisy
X_true = zeros(size(initial_states,1)*sampling_freq*t_end,n_state);
X_measured = zeros(size(initial_states,1)*sampling_freq*t_end,n_state);
U = zeros(size(initial_states,1)*sampling_freq*t_end,n_control);
X_dot = zeros(size(initial_states,1)*sampling_freq*t_end,n_state);
X_dot_true = zeros(size(initial_states,1)*sampling_freq*t_end,n_state); % estimation
T = zeros(size(initial_states,1)*sampling_freq*t_end,1);
data_size = 0;
fprintf("==> Total data size %e\n", size(X,1));

for i=1:size(initial_states,1)
    th1 = (rand(num_sine, 1)-0.5)*2*pi;
    th2 = (rand(num_sine, 1)-0.5)*2*pi;
    all_th1(i,:) = th1';
    all_th2(i,:) = th2';

    if rem(i,5) == 0
        fprintf("==> Simulating %03d out of %03d\n", i, size(initial_states,1));
    end

    x0 = initial_states(i,:)';
    qd = initial_states(i,1:2)';
    [T_tmp, X_true_tmp] = ode45(@eg3_TwoLinkArm_Dynamics,tspan,x0);
    X_measured_tmp = zeros([size(T_tmp,1),n_state]);
    U_tmp = zeros([size(T_tmp,1),n_control]);
    X_dot_true_tmp = zeros([size(T_tmp,1),n_state]);
    
    for j=1:size(T_tmp,1)
        t_tmp = T_tmp(j);
        x_true_tmp = X_true_tmp(j,:);
        x_measured_tmp = x_true_tmp + sqrt(var_x).*randn(1,n_state);
        x_estimated_tmp = x_true_tmp;
        u_tmp = eg3_TwoLinkArm_Controller(t_tmp, x_estimated_tmp);
        x_dot_true_tmp = eg3_TwoLinkArm_Dynamics_with_Input(t_tmp, x_true_tmp, u_tmp)';
        
        X_measured_tmp(j,:) = x_measured_tmp;
        U_tmp(j,:) = u_tmp;
        X_dot_true_tmp(j,:) = x_dot_true_tmp;
        
    end

    [b, a] = butter(2, 1e-1, 'low');
    X_processed_tmp = filtfilt(b, a, X_measured_tmp);
    X_dot_estimated_tmp = zeros([size(T_tmp,1), n_state]);
    
    for j=1:size(X_dot_estimated_tmp,1)
        switch j
            case 1
                % use FORWARD difference here for the first point
                X_dot_estimated_tmp(j,:) = (X_processed_tmp(j+1,:) - X_processed_tmp(j,:))*sampling_freq;
            case 2
                % use CENTRAL difference
                X_dot_estimated_tmp(j,:) = (X_processed_tmp(j+1,:) - X_processed_tmp(j-1,:))*sampling_freq/2;
            case size(X_dot_estimated_tmp,1) - 1
                % use CENTRAL difference
                X_dot_estimated_tmp(j,:) = (X_processed_tmp(j+1,:) - X_processed_tmp(j-1,:))*sampling_freq/2;
            case size(X_dot_estimated_tmp,1)
                % use BACKWARD difference here for the last point
                X_dot_estimated_tmp(j,:) = (X_processed_tmp(j,:) - X_processed_tmp(j-1,:))*sampling_freq;
            otherwise
                % fourth-order central difference method
                X_dot_estimated_tmp(j,:) = (-X_processed_tmp(j+2,:) + 8*X_processed_tmp(j+1,:)...
                    - 8*X_processed_tmp(j-1,:) + X_processed_tmp(j-2,:))*sampling_freq/12;
        end
    end

    % Save the values
    % Drop first 0.1 and last 0.1 second
    keep_ind = (T_tmp >= 0.1) & (T_tmp < t_end - 0.1);
    current_data_size = sum(keep_ind);
    X_true(data_size+1:data_size+current_data_size,:) = X_true_tmp(keep_ind,:);
    X(data_size+1:data_size+current_data_size,:) = X_processed_tmp(keep_ind,:);
    X_measured(data_size+1:data_size+current_data_size,:) = X_measured_tmp(keep_ind,:);
    X_dot_true(data_size+1:data_size+current_data_size,:) = X_dot_true_tmp(keep_ind,:);
    X_dot(data_size+1:data_size+current_data_size,:) = X_dot_estimated_tmp(keep_ind,:);
    U(data_size+1:data_size+current_data_size,:) = U_tmp(keep_ind,:);
    T(data_size+1:data_size+current_data_size,:) = T_tmp(keep_ind);
    data_size = data_size + current_data_size;

    % Plot x and measured x
    f = figure('visible','off');
    plot(T_tmp(keep_ind), X_true_tmp(keep_ind,1), '--', 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_true_tmp(keep_ind,2), '--', 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_true_tmp(keep_ind,3), '--', 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_true_tmp(keep_ind,4), '--', 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_measured_tmp(keep_ind,1), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_measured_tmp(keep_ind,2), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_measured_tmp(keep_ind,3), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_measured_tmp(keep_ind,4), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_processed_tmp(keep_ind,1), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_processed_tmp(keep_ind,2), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_processed_tmp(keep_ind,3), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_processed_tmp(keep_ind,4), 'LineWidth', 1);
    legend('theta1','theta2','dtheta1','dtheta2', ...
        'theta1_n','theta2_n','dtheta1_n','dtheta2_n', ...
        'theta1_e','theta2_e','dtheta1_e','dtheta2_e');
    set(gcf,'Position',[100 100 1000 500]);
    plot_name = sprintf('x_%03d.png', i);
    saveas(gcf, fullfile(result_dir, plot_name));
    close;
    
    % Plot x_dot and estimated x_dot
    f = figure('visible','off');
    plot(T_tmp(keep_ind), X_dot_true_tmp(keep_ind,1), '--', 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_dot_true_tmp(keep_ind,2), '--', 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_dot_true_tmp(keep_ind,3), '--', 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_dot_true_tmp(keep_ind,4), '--', 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_dot_estimated_tmp(keep_ind,1), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_dot_estimated_tmp(keep_ind,2), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_dot_estimated_tmp(keep_ind,3), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_dot_estimated_tmp(keep_ind,4), 'LineWidth', 1);
    legend('dtheta1','dtheta2','ddtheta1','ddtheta2', ...
        'dtheta1_e','dtheta2_e','ddtheta1_e','ddtheta2_e');
    set(gcf,'Position',[100 100 1000 500]);
    plot_name = sprintf('x_dot_%03d.png', i);
    saveas(gcf, fullfile(result_dir, plot_name));
    close;

end
%% Strip out extra data
fprintf("==> Total data size %e\n", data_size);
T = T(1:data_size);
X = X(1:data_size,:);
X_dot = X_dot(1:data_size,:);
U = U(1:data_size,:);
X_true = X_true(1:data_size,:);
X_measured = X_measured(1:data_size,:);
X_dot_true = X_dot_true(1:data_size,:);

%% Estimation error
error = X_dot_true - X_dot;
fprintf("==> Mean l2 error : \n");
disp(mean(vecnorm(error, 2, 2)));
fprintf("==> Max l2 error: \n");
disp(max(vecnorm(error, 2, 2)));
fprintf("==> Max component error: \n");
disp(max(abs(error)));

%% Save the data
fprintf("==> Saving data...\n");
save(fullfile(result_dir,'dataset'),'T','X','X_dot','U', ...
    'm_link1', 'm_motor1', 'I_link1', 'I_motor1', 'm_link2',...
    'm_motor2', 'I_link2', 'I_motor2', 'l1', 'l2', 'a1', 'a2',...
    'kr1', 'kr2', 'g', 'Fv1', 'Fv2', 'qd', 'Kp', 'Kd', ...
    'am1', 'am2', 'freq1', 'freq2', 'all_th1', 'all_th2');
save(fullfile(result_dir,'other'),'X_true', 'X_measured','X_dot_true', ...
    'm_link1', 'm_motor1', 'I_link1', 'I_motor1', 'm_link2',...
    'm_motor2', 'I_link2', 'I_motor2', 'l1', 'l2', 'a1', 'a2',...
    'kr1', 'kr2', 'g', 'Fv1', 'Fv2', 'qd', 'Kp', 'Kd', ...
    'am1', 'am2', 'freq1', 'freq2', 'all_th1', 'all_th2');
fprintf("==> Data saved. Finishing...\n");
