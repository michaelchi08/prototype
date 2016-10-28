function child = ga_point_mutation(x, prob)
    for i = 1:length(x)
        if (prob > rand)
            if (x(i) == 0)
                x(i) = 1;
            else
                x(i) = 0;
            end
        end
    end

    child = x;
end
