function [Y] = test_TSK_FS( test_data , pg, v, b)

% Calculate the predicted values of test data based on pg, v, b.
% 
% test_data:test dataset
% Y:predicted label

test_data = fromXtoZ(test_data,v,b);
Y = test_data*pg;
