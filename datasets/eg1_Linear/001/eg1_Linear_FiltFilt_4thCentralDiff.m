clear all;
rng('default');
filepath = fileparts(mfilename('fullpath'));
parts = strsplit(filepath, filesep);
parent_path = strjoin(parts(1:end-1), filesep);
result_dir = fullfile(parent_path, 'datasets', 'eg1_linear', '001');
rng(0);
copyfile(strcat(mfilename('fullpath'),'.m'),result_dir)

%% Define global parameters
n_state = 2;
n_control = 1; % actually, there is no control
A_sys = [-0.2, 2.0;
        -2.0, -0.2];
mu_noise = [0, 0];
sigma_noise = 0*eye(n_state);
try chol(sigma_noise, 'upper')
    sigma_noise_sqrt_upper = chol(sigma_noise, 'upper');
catch ME
    sigma_noise_sqrt_upper = zeros(n_state,n_state);
end

%% Gaussian noise variances
var_1 = 5e-5;
var_2 = 5e-5;
var_x = [var_1,var_2];

%% Simulation settings
t_end = 12.3;
sampling_freq = 100;
tspan = linspace(0, t_end, sampling_freq*t_end+1);
x_y_space = [-5, 5; -5, 5];
points_per_dim = [10, 10];
x1 = linspace(x_y_space(1,1), x_y_space(1,2), points_per_dim(1));
x2 = linspace(x_y_space(2,1), x_y_space(2,2), points_per_dim(2));
[X1,X2] = ndgrid(x1,x2);
initial_states = zeros(prod(points_per_dim),n_state);
initial_states(:,1) = reshape(X1,[],1);
initial_states(:,2) = reshape(X2,[],1);

%% Collect all trajectories
f0 = figure('visible','off');

%% Simulation
% simulated from ode45
X_true = zeros(size(initial_states,1)*sampling_freq*t_end,n_state);

% added measurement noise to X_true
X_measured = zeros(size(initial_states,1)*sampling_freq*t_end,n_state);

% smoothed from X_measured, used for training
X = zeros(size(initial_states,1)*sampling_freq*t_end,n_state);

% as it is
U = zeros(size(initial_states,1)*sampling_freq*t_end,n_control);

% one step dynamics based on X_true without process noise
X_dot_true = zeros(size(initial_states,1)*sampling_freq*t_end,n_state);

% calculated from num diff, used for training
X_dot = zeros(size(initial_states,1)*sampling_freq*t_end,n_state);

T = zeros(size(initial_states,1)*sampling_freq*t_end,1);
data_size = 0;
fprintf("==> Total data size %e\n", size(X,1));

for i=1:size(initial_states,1)
    if rem(i,5) == 0
        fprintf("==> Simulating %03d out of %03d\n", i, size(initial_states,1));
    end

    x0 = initial_states(i,:)';
    [T_tmp,X_true_tmp] = ode45(@(t,x) eg1_Linear_Dynamics(t,x,A_sys,mu_noise,sigma_noise_sqrt_upper),tspan,x0);
    X_measured_tmp = zeros([size(T_tmp,1),n_state]);
    U_tmp = zeros([size(T_tmp,1),n_control]);
    X_dot_true_tmp = zeros([size(T_tmp,1),n_state]);

    for j=1:size(T_tmp,1)
        t_tmp = T_tmp(j);
        x_true_tmp = X_true_tmp(j,:);
        x_measured_tmp = x_true_tmp + sqrt(var_x).*randn(1,n_state);
        x_dot_true_tmp = eg1_Linear_Dynamics(t_tmp,x_true_tmp',A_sys,zeros(1,n_state),zeros(n_state, n_state))';
        
        X_measured_tmp(j,:) = x_measured_tmp;
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
    keep_ind = (T_tmp >= 0.15) & (T_tmp < t_end - 0.15);
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
    plot(T_tmp(keep_ind), X_measured_tmp(keep_ind,1), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_measured_tmp(keep_ind,2), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_processed_tmp(keep_ind,1), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), X_processed_tmp(keep_ind,2), 'LineWidth', 1);
    legend('x','y','x_n','y_n','x_e','y_e');
    set(gcf,'Position',[100 100 1000 500]);
    plot_name = sprintf('x_%03d.png', i);
    saveas(gcf, fullfile(result_dir, plot_name));
    close;

    f = figure('visible','off');
    plot(X_true_tmp(keep_ind,1), X_true_tmp(keep_ind,2), '--', 'LineWidth', 1);
    hold on;
    plot(X_measured_tmp(keep_ind,1), X_measured_tmp(keep_ind,2), 'LineWidth', 1);
    hold on;
    plot(X_processed_tmp(keep_ind,1), X_processed_tmp(keep_ind,2), 'LineWidth', 1);
    legend('(x,y)','(x_n,y_n)','(x_e,y_e)');
    set(gcf,'Position',[100 100 1000 500]);
    plot_name = sprintf('xy_%03d.png', i);
    saveas(gcf, fullfile(result_dir, plot_name));
    close;
    
    % Plot x_dot and estimated x_dot
    f = figure('visible','off');
    plot(T_tmp(keep_ind), abs(X_dot_estimated_tmp(keep_ind,1) - X_dot_true_tmp(keep_ind,1)), 'LineWidth', 1);
    hold on;
    plot(T_tmp(keep_ind), abs(X_dot_estimated_tmp(keep_ind,2) - X_dot_true_tmp(keep_ind,2)), 'LineWidth', 1);
    legend('delta dx','delta dy');
    set(gcf,'Position',[100 100 1000 500]);
    plot_name = sprintf('x_dot_%03d.png', i);
    saveas(gcf, fullfile(result_dir, plot_name));
    close;
    
    % Collect x-y on one figure
    f0;
    plot(X_true_tmp(keep_ind,1), X_true_tmp(keep_ind,2), 'LineWidth', 1);
    hold on;
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

%% Save the figure
f0;
set(gcf,'Position',[0 0 500 500]);
axis equal;
saveas(gcf, fullfile(result_dir, 'xy_collect.png'));

%% Save the data
fprintf("==> Saving data...\n");
save(fullfile(result_dir,'dataset'),'T','X','X_dot','U', ...
    'A_sys','mu_noise','sigma_noise','sigma_noise_sqrt_upper');
save(fullfile(result_dir,'other'),'X_true','X_measured','X_dot_true', ...
    'A_sys','mu_noise','sigma_noise','sigma_noise_sqrt_upper');
fprintf("==> Data saved. Finishing...\n");