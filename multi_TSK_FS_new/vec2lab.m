function [ labels ] = vec2lab( vectors )

% convert the vector(one of hot) of the label to a scalar



[~,labels]=max(vectors,[],2);

end

