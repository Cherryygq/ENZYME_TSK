
% load and preprocess dataset
% construction the multi-view TSK fuzzy system
% Calculate the performance of the constructed system

clear;
clc;
data_num = 6;
folds_num = 5;
mean_result = zeros(folds_num, 4);
result = zeros(2,4);
for k = 1:folds_num
    load(['../data/feature/fold_' num2str(k) '/data_train.mat']);
    load(['../data/feature/fold_' num2str(k) '/data_predict.mat']);
    mulview_tr_cell = {tr_X_1; tr_X_2};
    mulview_te_cell = {te_X_1; te_X_2};
%     load(['../data/feature/fold_' num2str(k) '/data_train.mat']);
%     load(['../data/feature/fold_' num2str(k) '/data_predict.mat']);
%     mulview_tr_cell = {tr_X_2};
%     mulview_te_cell = {te_X_2};
    [ best_acc_result, TSK_result ] = expt_mul_TSK( mulview_tr_cell, mulview_te_cell, tr_Y, te_Y, k);
    mean_result(k,1) = mean_result(k,1) + best_acc_result.acc_mean;
    mean_result(k,2) = mean_result(k,2) + best_acc_result.pre_mean;
    mean_result(k,3) = mean_result(k,3) + best_acc_result.rec_mean;
    mean_result(k,4) = mean_result(k,4) + best_acc_result.f1_mean;
end
result(1,:) = mean(mean_result);
save(['../data/result/data' num2str(data_num) '_result.mat'], 'result', 'mean_result');
