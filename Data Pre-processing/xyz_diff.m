function [maA,saA ] = xyz_diff( a0) % m111,m121,m211,m221,m112,m122,m212,m222,s111,s121,s211,s221,s112,s122,s212,s222,
A = zeros(11616*3,2400); Ax = zeros(11616,2400);Ay = zeros(11616,2400);Az = zeros(11616,2400);
%m = 11616;

for i = 1:96
    for j = 1:31
        str = sprintf('%d-%d.xyz',i,j-1);
        [b a(:,1) a(:,2) a(:,3)] = textread(str,['%s' '%f' '%f' '%f']);
        a_both = [a a0];a1 = abs(a_both(:,1:3)-a_both(:,4:6)); 
        Acx = -54.96128*(a1(:,1)>27.48); Acy = -23.798928*2*(a1(:,2)>23.8); Acz = -39.8142*(a1(:,3)>19.91);
        a1 = a1 + [Acx Acy Acz];
        a1 = abs(a1);
%         ax = sortrows(a_both,-1); ax1 = ax(1:m/2);ax2 = ax(m/2+1:end);
%         ax1y = sortrows(ax1,-2); ax1y1 = ax1y(1:m/4); ax1y2 = ax1y(m/4+1:end);
%         ax2y = sortrows(ax2,-2); ax2y1 = ax2y(1:m/4); ax2y2 = ax2y(m/4+1:end);
%         ax1y1z = sortrows(ax1y1,-3); ax1y1z1 = ax1y1z(1:m/8); ax1y1z2 = ax1y1z(m/8+1:end);
%         ax1y2z = sortrows(ax1y2,-3); ax1y2z1 = ax1y2z(1:m/8); ax1y2z2 = ax1y2z(m/8+1:end);
%         ax2y1z = sortrows(ax2y1,-3); ax2y1z1 = ax2y1z(1:m/8); ax2y1z2 = ax2y1z(m/8+1:end);
%         ax2y2z = sortrows(ax2y2,-3); ax2y2z1 = ax2y2z(1:m/8); ax2y2z2 = ax2y2z(m/8+1:end);
%         a1x1y1z1 = abs(ax1y1z1(:,1:3)-ax1y1z1(:,4:6)); a1x1y1z2 = abs(ax1y1z2(:,1:3)-ax1y1z2(:,4:6));
%         a1x1y2z1 = abs(ax1y2z1(:,1:3)-ax1y2z1(:,4:6)); a1x1y2z2 = abs(ax1y2z2(:,1:3)-ax1y2z2(:,4:6));
%         a1x2y1z1 = abs(ax2y1z1(:,1:3)-ax2y1z1(:,4:6)); a1x2y1z2 = abs(ax2y1z2(:,1:3)-ax2y1z2(:,4:6));
%         a1x2y2z1 = abs(ax2y2z1(:,1:3)-ax2y2z1(:,4:6)); a1x2y2z2 = abs(ax2y2z2(:,1:3)-ax2y2z2(:,4:6));
% 
%         A1x1y1z1(:,(i-1)*30+j) = a1x1y1z1(:); A1x1y1z2(:,(i-1)*30+j) = a1x1y1z2(:);
%         A1x1y2z1(:,(i-1)*30+j) = a1x1y2z1(:); A1x1y2z2(:,(i-1)*30+j) = a1x1y2z2(:);
%         A1x2y1z1(:,(i-1)*30+j) = a1x2y1z1(:); A1x2y1z2(:,(i-1)*30+j) = a1x2y1z2(:);
%         A1x2y2z1(:,(i-1)*30+j) = a1x2y2z1(:); A1x2y2z2(:,(i-1)*30+j) = a1x2y2z2(:);
%         
        A(:,(i-1)*31+j) = a1(:);
        Ax(:,(i-1)*31+j) = a1(:,1);
        Ay(:,(i-1)*31+j) = a1(:,2);
        Az(:,(i-1)*31+j) = a1(:,3);
        
    end
    i
end
% m111 = mean(A1x1y1z1)'; m112 = mean(A1x1y1z2)'; s111 = std(A1x1y1z1)'; s112 = std(A1x1y1z2)';
% m121 = mean(A1x1y2z1)'; m122 = mean(A1x1y2z2)'; s121 = std(A1x1y2z1)'; s122 = std(A1x1y2z2)';
% m211 = mean(A1x2y1z1)'; m212 = mean(A1x2y1z2)'; s211 = std(A1x2y1z1)'; s212 = std(A1x2y1z2)';
% m221 = mean(A1x2y2z1)'; m222 = mean(A1x2y2z2)'; s221 = std(A1x2y2z1)'; s222 = std(A1x2y2z2)';
maA = mean(abs(A))'; saA = std(abs(A))';
%maAx = mean(abs(Ax))'; saAx = std(abs(Ax))';
%maAy = mean(abs(Ay))'; saAy = std(abs(Ay))';
%maAz = mean(abs(Az))'; saAz = std(abs(Az))';
end

