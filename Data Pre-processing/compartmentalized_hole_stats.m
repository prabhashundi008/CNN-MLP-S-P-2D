function [ S3c,St3c,S1c,St1c ] = compartmentalized_hole_stats(  ) %,l1i,l2i,l3i,l1e,l2e,l3e
 
Allind = (1:11616)';
% f = find_holes(1,bins);
% [l1,l2,l3] = size(f);
% p = l1/l1i+1; q = l2/l2i+1; r = l3/l3i+1;
% [p q r]
% S = zeros(p*q*r,1); St = S;
% for mi = 1:m
% str = sprintf('\nPercentage Complete: %0.2f\n', mi/m*100);
% fprintf(str);
%     f = find_holes(mi,bins);
%     for i = 1:p-1
%         X = ((i-1)*l1i+1:i*l1i)';
%         for j = 1:q-1
%              Y = ((j-1)*l2i+1:j*l2i)';
%             for k = 1:r-1 
%                 Z = ((k-1)*l3i+1:k*l3i)';
%                 f1 = f(X,Y,Z);
%                 S((i-1)*p^2+(j-1)*q+k,1) = sum(f1(:));
%                 St((i-1)*p^2+(j-1)*q+k,1) = std(f1(:));
%             end
%             Z = (l3-l3e+1:l3)';
%             f1 = f(X,Y,Z);
%             S((i-1)*p^2+(j-1)*q+r,1) = sum(f1(:));
%             St((i-1)*p^2+(j-1)*q+r,1) = std(f1(:));
%         end
%         Y = (l2-l2e+1:l2)';
%         for k = 1:r-1 
%             Z = ((k-1)*l3i+1:k*l3i)';
%             f1 = f(X,Y,Z);
%             S((i-1)*p^2+(q-1)*q+k,1) = sum(f1(:));
%             St((i-1)*p^2+(q-1)*q+k,1) = std(f1(:));
%         end
%         Z = (l3-l3e+1:l3)';
%         f1 = f(X,Y,Z);
%         S((i-1)*p^2+(q-1)*q+r,1) = sum(f1(:));
%         St((i-1)*p^2+(q-1)*q+r,1) = std(f1(:));
%     end
%     X = (l1-l1e+1:l1)';
%     for j = 1:q-1
%              Y = ((j-1)*l2i+1:j*l2i)';
%             for k = 1:r-1 
%                 Z = ((k-1)*l3i+1:k*l3i)';
%                 f1 = f(X,Y,Z);
%                 S((p-1)*p^2+(j-1)*q+k,1) = sum(f1(:));
%                 St((p-1)*p^2+(j-1)*q+k,1) = std(f1(:));
%             end
%             Z = (l3-l3e+1:l3)';
%             f1 = f(X,Y,Z);
%             S((p-1)*p^2+(j-1)*q+r,1) = sum(f1(:));
%             St((p-1)*p^2+(j-1)*q+r,1) = std(f1(:));
%     end
%     Y = (l2-l2e+1:l2)';
%     for k = 1:r-1 
%         Z = ((k-1)*l3i+1:k*l3i)';
%         f1 = f(X,Y,Z);
%         S((p-1)*p^2+(q-1)*q+k,1) = sum(f1(:));
%         St((p-1)*p^2+(q-1)*q+k,1) = std(f1(:));
%     end
%     Z = (l3-l3e+1:l3)';
%     f1 = f(X,Y,Z);
%     S((p-1)*p^2+(q-1)*q+r,1) = sum(f1(:));
%     St((p-1)*p^2+(q-1)*q+r,1) = std(f1(:));
% 
% end
S3 = zeros(2976,27); St3 = S3; S1 = zeros(2976,1); St1 = S1;
for ii = 1:96
    for jj = 1:31

SS = zeros(27,1); SSt = zeros(27,1);

h = find_curved_holes(ii,jj-1,Allind);
SS1 = sum(h(:)); SSt1 = std(h(:));
for i = 1:3
    for j = 1:3
        for k = 1:3
            [x,y,z] = xyz_ranges(i,j,k);
            h1 = h(x,y,z);
            SS((i-1)*9+(j-1)*3+k,1) = sum(h1(:));
            SSt((i-1)*9+(j-1)*3+k,1) = std(h1(:));
        end
    end
end
S3((ii-1)*31+jj,:) = SS'; St3((ii-1)*31+jj,:) = SSt';
S1((ii-1)*31+jj,1) = SS1; St1((ii-1)*31+jj,1) = SSt1;
[ii jj-1]
    end
end
S3c = S3;
St3c = St3;
S1c = S1;
St1c = St1;
end

