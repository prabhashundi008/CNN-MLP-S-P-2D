function [holes] = find_curved_holes(ii,jj,Bind)
bins = make_curved_bins(ii,jj,Bind);
[p,q,r] = size(bins);
holes = zeros(p-2,q-2,r-2);
cen = 0; fc = 1; ec = 0; cor = 0;
for i = 2:p-1
    for j = 2:q-1
        for k = 2:r-1
            if bins(i,j,k) == 0
                 holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cen;
                 if bins(i-1,j,k) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + fc;
                 end
                 if bins(i+1,j,k) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + fc;
                 end
                 if bins(i,j-1,k) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + fc;
                 end
                 if bins(i,j+1,k) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + fc;
                 end
                 if bins(i,j,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + fc;
                 end
                 if bins(i,j,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + fc;
                 end
                 if bins(i-1,j-1,k) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i-1,j+1,k) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i+1,j-1,k) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i+1,j+1,k) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i-1,j,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i-1,j,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i+1,j,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i+1,j,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i,j-1,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i,j-1,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i,j+1,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i,j+1,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + ec;
                 end
                 if bins(i-1,j-1,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cor;
                 end
                 if bins(i-1,j-1,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cor;
                 end
                 if bins(i-1,j+1,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cor;
                 end
                 if bins(i-1,j+1,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cor;
                 end
                 if bins(i+1,j-1,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cor;
                 end
                 if bins(i+1,j-1,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cor;
                 end
                 if bins(i+1,j+1,k-1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cor;
                 end
                 if bins(i+1,j+1,k+1) == 0
                    holes(i-1,j-1,k-1) = holes(i-1,j-1,k-1) + cor;
                 end
            end  
        end
    end
end
end

