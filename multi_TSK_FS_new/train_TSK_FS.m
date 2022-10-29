function [best_pg, best_v, best_b, best_TSK_result] = train_TSK_FS( tr_data, te_data, tr_label, te_label, folds_num, view_num, k)

% train classifier of each view

Ms = [4:1:7];
lamdas = [0,2.^(-5:5)];
a = 0;
best_acc_mean = 0;
for lamda = lamdas
    a = a + 1;
    c = 0;
    for M = Ms
        c = c + 1;
        result = zeros(folds_num,4);
        for fold=1:folds_num
            [v,b] = preproc(tr_data, M);
            Xg = fromXtoZ(tr_data,v,b);   %Xg:N*K
            Xg1 = Xg'*Xg;
            pg = pinv(Xg1 + lamda*eye( size(Xg1)))*Xg'*tr_label;    %Solving the consequent parameters of the TSK-FS
            [te_Y] = test_TSK_FS( te_data , pg, v, b);
            te_Y = vec2lab(te_Y);
            [ acc, pre, recall, f1 ] =  confusion_matrix(te_label, te_Y );
            result(fold,1)=acc;
            result(fold,2)=pre;
            result(fold,3)=recall;
            result(fold,4)=f1;
        end
        acc_te_mean = mean(result(:,1));
        pre_te_mean = mean(result(:,2));
        rec_te_mean = mean(result(:,3));
        f1_te_mean = mean(result(:,4));
        if acc_te_mean>best_acc_mean
			best_acc_mean = acc_te_mean;
            best_TSK_result.acc = acc_te_mean;
            best_TSK_result.pre = pre_te_mean;
            best_TSK_result.rec = rec_te_mean;
            best_TSK_result.f1 = f1_te_mean;
            best_pg = pg;
            best_v = v;
            best_b = b;
        end
        fprintf('train TSK FS:%d/5------view:%d\nNumber of iterations:%d/%d------%d/%d\n', k, view_num, a, size(lamdas,2), c, size(Ms,2));
        fprintf('best acc result:\nacc:%.4f  pre:%.4f  rec:%.4f  f1:%.4f\n\n',best_TSK_result.acc, best_TSK_result.pre, best_TSK_result.rec, best_TSK_result.f1);
    end %end M
end %end lamda
