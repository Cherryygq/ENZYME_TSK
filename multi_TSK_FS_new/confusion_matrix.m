function [ acc, pre, rec, f1 ] = ...
    confusion_matrix(labels, te_Y )

% Calculate the confusion matrix of the data set based on the predicted values
% Calculate performance indicators based on the confusion matrix
% 
% labels:real label
% te_Y:predicted label
% accuracy, sensitivity, specificity:three performance indicators

    conf_mat = confusionmat(labels, te_Y);
%     disp(conf_mat);


    acc = sum(diag(conf_mat))/size(labels,1);
%     disp(acc);

    conf_mat1 = bsxfun(@rdivide, conf_mat, (sum(conf_mat, 1)));
    pre = mean(diag(conf_mat1));
%     disp(pre);

    conf_mat2 = bsxfun(@rdivide, conf_mat, (sum(conf_mat, 2)));
    rec= mean(diag(conf_mat2));
%     disp(rec);

    f1 = 2*pre*rec/(pre+rec);
%     disp(f1);

end
