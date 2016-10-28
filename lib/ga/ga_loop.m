function ga_loop(max_generations, bitstring_length, population_size, eval_func)
    generation = 0;
    population = ga_generation_population(bitstring_length, population_size);
    scores = zeros(1, population_size);

    while (max_generations != generation)
        population = ga_reproduce(population, scores, c_prob, m_prob, tournament_size, cmp)
    end
end
