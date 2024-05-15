function [ mat ] = definirZone( matIn )
%DEFINIRZONE Summary of this function goes here
%   Detailed explanation goes here
n = size(matIn,1);
m = size(matIn,2);
testN = zeros(n,1);
testM = zeros(1,m);
bool = 1;
mat = matIn;
while bool
    j = 1;
    while bool && j <= m
        bool = mat(1,j) == 1;
        j = j+1;
    end
    if bool
        mat(1,:) = [];
    end
end

bool = 1;
while bool
    n = size(mat,1);
    bool = 1;
    j = 1;
    while bool && j <= m
        bool = mat(n,j) == 1;
        j = j+1;
    end
    if bool
        mat(n,:) = [];
    end
end

n = size(mat,1);
bool = 1;
while bool
    m = size(mat,2);
    bool = 1;
    j = 1;
    while bool && j <= n
        bool = mat(j,m) == 1;
        j = j+1;
    end
    if bool
        mat(:,m) = [];
    end
end

bool = 1;
while bool
    bool = 1;
    j = 1;
    while bool && j <= n
        bool = mat(j,1) == 1;
        j = j+1;
    end
    if bool
        mat(:,1) = [];
    end
end

end

