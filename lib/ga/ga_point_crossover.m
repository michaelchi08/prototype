function [child_1, child_2] = ga_point_crossover(x, y, prob)
    % choose crossover point
    xo_index = floor(rand * length(x));
    if (xo_index == 0)
        xo_index = 1;
    end

    % perform crossover
    xo_index
    child_1 = x;
    child_1(xo_index:end) = y(xo_index:end);
    child_2 = y;
    child_2(xo_index:end) = x(xo_index:end);
end
