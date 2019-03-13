function [bins, Z] = make_curved_bins(ii,jj,Bind)
sprintf('[%d  %d]',ii,jj);
str = sprintf('%d-%d.xyz',ii,jj);
if ii == 0
    str = ('0-dump.0.xyz');
end
[b a(:,1) a(:,2) a(:,3)] = textread(str,['%s' '%f' '%f' '%f']);
b1 = a(Bind,:);
Xedges = (-0.62456:1.24912:54.33672 )';
Yedges = (-25.2413:2.163541:24.53012 )';
Z = zeros(11,23);
for i = 1:11
for j = 0:21
a1 = a(968*i+1:968*(i+1),:);
a1 = a1(44*j+1:44*(j+1),:);
Z(i,j+1) = median(a1(:,3));
end
end
Z0 = [1*ones(1,size(Z,2)); Z];
Z1 = [Z; 39*ones(1,size(Z,2))];
Ze = (Z0 + Z1)/2;
%Zedges = (-0.6553:3.16345:37.3061)';
bins = zeros(size(Xedges,1)-1,size(Yedges,1)-1,size(Ze,1)-1);

for pin = 1:size(b1,1)
    cc = b1(pin,:)';
    for i = 1:size(Xedges,1)-1
        high = Xedges(i+1,1); low = Xedges(i,1);
        if cc(1) <= high && cc(1) >= low
           for j = 1:size(Yedges,1)-1
               Zedges = Ze(:,j);
               high = Yedges(j+1,1); low = Yedges(j,1);
               if cc(2) <= high && cc(2) >= low 
                  for k = 1:size(Zedges,1)-1
                      high = Zedges(k+1,1); low = Zedges(k,1);
                      if cc(3) <= high && cc(3) >= low
                         bins(i,j,k) = bins(i,j,k) + 1;
                      end
                  end
               end
           end
        end
    end
end
bins = bins(2:end,2:end-1,:);
end

