%a = 2.^(-5:5);
%disp(a);



A=[2:1:6];
disp(A);

% load('../data/result/data3_result_new1000.mat');
load('data6_result.mat');
disp(mean_result);
disp(result);


% clear;
% clc;
% %data_num = 1;
% folds_num = 5;
% a = 0;
% b=0;
% c=0;
% d=0;
% e=0;
% f=0;
% 
% 
% for k = 1:folds_num
%     load(['../data/feature/fold_' num2str(k) '/data_train.mat']);
%     load(['../data/feature/fold_' num2str(k) '/data_predict.mat']);
%     %mulview_tr_cell = {tr_X_1; tr_X_2};
%     %mulview_te_cell = {te_X_1; te_X_2};
%     %[ best_acc_result, TSK_result ] = expt_mul_TSK( mulview_tr_cell, mulview_te_cell, tr_Y, te_Y, k);
% %     a = a+numel(find(isnan(tr_X_1)));
% %     b = b+numel(find(isnan(tr_X_2)));
% %     c = c+numel(find(isnan(te_X_1)));
% %     d = d+numel(find(isnan(te_X_2)));
% %     e = e+numel(find(isnan(tr_Y)));
% %     f = f+numel(find(isnan(te_Y)));
%     a = numel(find(isinf(tr_X_1)));
%     b = numel(find(isinf(tr_X_2)));
%     c = numel(find(isinf(te_X_1)));
%     d = numel(find(isinf(te_X_2)));
%     e = numel(find(isinf(tr_Y)));
%     f = numel(find(isinf(te_Y)));
%    
%     
% end
% disp(num2str(a));
% disp(num2str(b));
% disp(num2str(c));
% disp(num2str(d));
% disp(num2str(e));
% disp(num2str(f));

% y_true = [1, 1, 2, 4, 2, 1, 5, 3, 2, 1];
% y_pred = [1, 2, 1, 5, 2, 3, 5, 2, 1, 1];
% conf_mat = confusionmat(y_true, y_pred);
% disp(conf_mat);
% 
% 
% acc = sum(diag(conf_mat))/sum(y_true);
% mm = size(y_true,2);
% disp(mm);
% 
% conf_mat1 = bsxfun(@rdivide, conf_mat, sum(conf_mat, 2));
% % disp(conf_mat1);
% pre = mean(diag(conf_mat1));
% disp(pre);
% 
% conf_mat2 = bsxfun(@rdivide, conf_mat, (sum(conf_mat, 1)+0.0001));
% % disp(conf_mat2);
% recall = mean(diag(conf_mat2));
% disp(recall);
% 
% f1 = 2*pre*recall/(pre+recall);
% disp(f1);

% confus,accuracy,numcorrect,precision,recall1,F = try2 (y_true,y_pred);
% disp(accuracy);
% disp(precision);
% disp(recall1);
% disp(F);
% pre1 = mean(diag(conf_mat)./(sum(conf_mat,2)+0.0001));
% disp(pre1);
% 
% rec1 = mean(diag(conf_mat)./(sum(conf_mat,1)+0.0001));
% disp(rec1);
