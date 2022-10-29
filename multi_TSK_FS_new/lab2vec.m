function [ vectors ] = lab2vec( labels )

% convert the scalar of the label to a vector(one of hot)

N=length(labels);
M=max(labels);

vectors=zeros(N,M);
for m=1:M
    indices=(labels==m);
    vectors(indices,m)=1;
end

end

