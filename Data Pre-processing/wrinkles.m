function [ L ] = wrinkles( ii,jj )

sprintf('[%d  %d]',ii,jj);
str = sprintf('%d-%d.xyz',ii,jj);
if ii == 0
    str = ('0-dump.0.xyz');
end
[b a(:,1) a(:,2) a(:,3)] = textread(str,['%s' '%f' '%f' '%f']);

L = zeros(11,1);
for l = 1:11
    L(l) = std(a(968*l+1:(l+1)*968,3));
end
L = mean(L);
