function [ res ] = densite( M, i1, j1, i2, j2)
%DENSITE Summary of this function goes here
%   Detailed explanation goes here
dt = (i2-i1+1)*(j2-j1+1);
res = 0;
for i=i1:i2
    for j=j1:j2
        res = res+M(i,j);
    end
end
res = res/dt;

end

