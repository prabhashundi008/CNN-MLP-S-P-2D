function [train,valid,test  ] = write_train_test_files( T,Y,tr,va,te )

tr1 = (1:tr)'; va1 = (tr+1:tr+va)'; te1 = (tr+va+1:tr+va+te)';
train = tr1; valid = va1; test = te1;

for i = 1:79
    train = [train; 31*i+tr1];
    valid = [valid; 31*i+va1];
    test = [test; 31*i+te1];
end

X_train = T(train,:); X_test = T(valid,:); X_test2 = T(test,:);
Y_train = Y(train,1); Y_test = Y(valid,1); Y_test2 = Y(test,1);
dlmwrite('X_train.txt',X_train,'delimiter','\t')
dlmwrite('X_test.txt',X_test,'delimiter','\t')
dlmwrite('Y_test.txt',Y_test,'delimiter','\t')
dlmwrite('Y_train.txt',Y_train,'delimiter','\t')
dlmwrite('X_test2.txt',X_test2,'delimiter','\t')
dlmwrite('Y_test2.txt',Y_test2,'delimiter','\t')

end

