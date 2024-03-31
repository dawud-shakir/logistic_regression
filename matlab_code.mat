% logistic regression with (and without) pca

clc;
clearvars;
close all;

sim.plot_things.save_figures = 0


%sim.in_csv = '../in/mfcc_13_labels.csv'; 
%sim.in_csv = '../in/mfcc_40_labels.csv'; 

%sim.in_csv = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/mfcc_40_labels.csv";
sim.in_csv = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/mfcc_13_labels.csv";

sim.coeffs = 1:13; % set these: [1,end], 1:40, [1, 2, 5, 13], ...

sim.num_coeffs = length(sim.coeffs);

% hyper params
sim.penalty = 0.001;
sim.learning_rate = 0.001;
sim.iterations = 1000;      

% sampling
sim.sampling.with_replacement = 1; %
sim.train_ratio = 0.80;

sim

% for kaggle ...
%{
    T = table(column_id', column_class', 'VariableNames', {'id', 'class'});
    
    out_file = "out.csv";
    if exist(out_file, 'file') == 2
        error(sprintf('file error: %s already exists!', out_file));
    else
        disp(['outfile: ', out_file])
        writetable(T, out_file)
    end
%}


accuracies = [];    


total_runs = sim.num_coeffs

for i = 0:total_runs
    sim.pca_count = i;   % this many eigens

    sim  % show
    
    old_seed = rng(0); % same seed for each run
    
    [loss, accuracy] = train_and_test(sim);
    accuracies = [accuracies, accuracy];    
    
    hold on
    
            subplot(ceil(sqrt(total_runs)), ceil(sqrt(total_runs)), 1 + i)  % add plot to figure
        

                
            if sim.pca_count > 0
                with_pca_title = ["Accuracy (with PCA)" num2str(sim.pca_count)];
                line_color = 'r';
            else
                with_pca_title = ["Accuracy (without PCA)" num2str(0)];
                line_color = 'b';
            end
                
                
             plot(accuracy, line_color, 'LineWidth', 1.5), 
                title(with_pca_title), 
                xlabel('Iteration'), 
                xlim([0,sim.iterations]), 
                ylim([0,1]),
                grid on,
    hold off

end

accuracies  

disp('best ever='), max(max(accuracies))
disp('worst ever='), min(min(accuracies))


summary(dataset(accuracies))





function [W, accuracies, losses] = train_and_test(sim)

labels = {'blues', 'classical', 'country', 'disco', 'hiphop', ...
          'jazz', 'metal', 'pop', 'reggae', 'rock'};



% load training data
table = readtable(sim.in_csv);


X = table2array(table(:, sim.coeffs));
%X = table2array(table(:, 1:13));
Y_labels = table2array(table(:, end));



if sim.pca_count>0  
 % with pca
    

    % 1. Center columns
    X2 = X - mean(X, 1); 
    
    % 2. Build covariance matrix
    cov_mat = cov(X2); % 13 x 13
    
    % 3. Eigenvectors and eigenvalues
    [eig_vecs, eig_vals] = eig(cov_mat);
    
    % 4. Sort eigenvectors highest to lowest
    [~,idx] = sort(diag(eig_vals), 'descend');
    eig_vecs_sorted = eig_vecs(:,idx);
    
    % 5. Use the top k eigenvectors
    
    k=sim.pca_count;
    projection_mat = eig_vecs_sorted(:,1:k);
    
        % cumsum >= 0.95:
        [~, ~, ~, ~, variance_ratios_explained, ~] = pca(X);
        cumulative = cumsum(variance_ratios_explained); 
        [~, argmax] = max(cumulative >= 95, [], 1);  
        
        % capture 95% variance of all columns with this many pca's:
        d = argmax + 1;         % +1 not for matlab indexing: argmax+1 used in python version too     
        
        

    % 6. Project onto centered columns to reduce
    X_reduced = X2 * projection_mat;
    
    X = X_reduced;

end


% standardize features
X = (X - mean(X)) ./ std(X);  
% same: X = zscore(X);  

% one hot encode labels
num_labels = size(unique(Y_labels), 1);
num_samples = size(Y_labels, 1);

% 1-hot encode
Y = zeros(num_labels, num_samples);
for i = 1:num_samples
    Y(strcmp(labels, Y_labels{i}), i) = 1;
end

% column of ones
X = [ones(num_samples, 1) X];

% split data into training and testing sets
num_train = round(sim.train_ratio * num_samples);

if sim.sampling.with_replacement==1
    % randomly shuffle indices with repetitions (with replacement)
    indices = randi(size(X, 1), 1, size(X, 1));

else
    % randomly shuffle indices without repetitions (without replacement) 
    indices = randperm(size(X, 1));
end

% split
X_train = X(indices(1:num_train), :);
X_test = X(indices(num_train+1:end), :);
Y_train = Y(:,indices(1:num_train));
Y_test = Y(:,indices(num_train+1:end));             

W = rand(num_labels, size(X_train, 2));    % only random

W(:,1) = 1/9;  % bias

sigmoid = @(s) 1./(1 + exp(-s));

losses = zeros(sim.iterations, 1);
accuracies = zeros(sim.iterations, 1);


for j = 1:sim.iterations

    
    PY = sigmoid(W*X_train'); 
    
    W = W + sim.learning_rate * ((Y_train - PY) * X_train - sim.penalty * W);

    % Make prediction
    
    [~, argmax] = max(sigmoid(W*X_test'), [], 1);   % row max
    
   
    score = 0;
    for i = 1:size(Y_test, 2)
        % Y_test is one hot encoded
        % argmax are numbers

        if Y_test(argmax(i), i) == 1

            score = score + 1;
        end
    end
    
    accuracy = score / size(Y_test, 2)
    loss = mean(mean(Y_train - PY))         

    accuracies(j) = accuracy;
    losses(j) = loss;           % less loss = better guess
end
end
