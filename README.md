# Heartbeat_Hackers_code

# This program was created by Gabriella Flinn, Jonathan French, and Kruze Peacock to classify EKG data using a decision tree. 
# It was a part of our final project for BME 3053C Computer Applications for Biomedical Engineerings

# Header, credit, and contact information
% EKG analysis program for BME3053C final project
% BME3053C Heartbeat Hackers Final Project
% Author: Gabriella Flinn, Jonathan French, Kruze Peacock
% Course: BME 3053c Computer Application for BME 
% Term: Fall 2023
% J. Crayton Pruitt Family Department of Biomedical Engineering 
% University of Florida
% Email: flinngabriella@ufl.edu 
% November 26, 2023. 

# Open data: we are using the 'ptbdb_abnormal.csv' and 'ptbdb_normal.csv' EKG files as testing and training data
# To retrieve the files, the 'Heartbeat_Hackers_data(0).zip' and 'Heartbeat_Hackers_data(1).zip' must be unzipped and the resulting files must be in the pathway. 
clc; clear;
abnormal_data=readmatrix('ptbdb_abnormal.csv'); % load the normal and abnormal training data  
normal_data=readmatrix("ptbdb_normal.csv");

# Visualize data using the mean and plot functions. We also used Fourier Transforms to further visualize the data
mean_abnormal=mean(abnormal_data); %Find the mean of the abnormal data 
mean_normal=mean(normal_data); %Find the mean of the normal data 
x=1:188; %Create x-axis (that correlates to the vector length of normal abnormal data
figure(1)
plot(x, mean_abnormal) %Plot mean of abnormal data 
hold on 
plot(x, mean_normal) %Plot mean of normal data on the same graph 
title('Mean of Abnormal and Normal ECG Samples')
legend('Abnormal','Normal')
xlabel('Number of Samples')
ylabel('Mean')

ft_abnormal=abs(fft(abnormal_data)); %Fourier transform of abnormal data 
ft_normal=abs(fft(normal_data)); %Fourier transform of normal dat 
fs=360; %Sampling frequency 
L=188; %Number of samples 
Hz=(0:L-1)/L*fs;
figure(2)
plot(Hz,ft_abnormal)
hold on
plot(Hz, ft_normal)
title('Fourier Transform of Abnormal and Normal ')
xlabel('Fequency (Hz)')
ylabel('Power')
legend('Abnormal','Normal')

# Concatenate Abnormal and Normal Data to make the 'total_data" variable containing all of the data. This is also where we split the data into the training and testing datasets  
total_data=[abnormal_data; normal_data];
labels=total_data(:,188); %Labels found in the last row 1=Abnormal; 0=Normal

% Split Train:Test=70:30
indices=crossvalind('HoldOut',size(total_data,1),0.3);

# Create a Decision Tree Matrix using fitctree function (takes a few minutes to create since the data files are large)
rng(101); %Set seed to 101 for reproducibility 
cv=cvpartition(size(total_data,1), 'Holdout', .3);
id_training=cv.training;
x_train=total_data(id_training, 1:187); %Does not include the label (last) column
y_train=labels(id_training);
x_test=total_data(~id_training, 1:187); %Does not include the label (last) column
y_test=labels(~id_training);
dtree=fitctree(x_train,y_train, 'CategoricalPredictors', 'all', 'MinLeafSize', 1, 'MaxNumSplits', 100, 'SplitCriterion', 'deviance');
view(dtree);

# Visualize the Decision Matrix using heatmap function 
y_predict=predict(dtree,x_test); %The predicted labels 
Confusion=confusionmat(y_test,y_predict);
disp(Confusion)
Confusion_normalized=Confusion/sum(Confusion, 'all');
class_labels={'Abnormal', 'Normal'};
figure(3)
heatmap(Confusion_normalized, 'XLabel', 'Predicted Label', 'YLabel', 'True Label', 'ColorbarVisible', 'on', 'XDisplayLabels', class_labels, 'YDisplayLabels', class_labels, 'Colormap', parula, 'FontSize', 8, 'CellLabelColor', 'none');
accuracy = sum(y_predict == total_data(~id_training, end)) / numel(total_data(~id_training, end));
fprintf("accuracy is "+num2str(accuracy))

# run an example ekg and display it. Choose a specific index to view the EKG associated with it as well as see what the decision tree's prediction and true labels
example_index = 1; % Change this to the index you want to test
example_data = total_data(example_index, 1:end-1); % Exclude the label column
true_label = labels(example_index);
predicted_label = predict(dtree, example_data);
disp(['Example EKG index: ', num2str(example_index)]);
disp('A label of 0 refers to a Normal EKG, whhile a label of 1 refers to an abnormal EKG');
disp(['True Example Label: ', num2str(true_label)]);
disp(['Predicted Example Label: ', num2str(predicted_label)]);

figure;
plot(example_data)
title('example ekg')
xlabel('Time');
ylabel('Amplitude');
