function p = ga_generate_popoulation(bitstring_length, population_size)
    p = zeros(1, bitstring_length, population_size);

    for i = 1:population_size
        for j = 1:bitstring_length
            p(1, j, i) = rand > 0.5;
        end
    end
end
