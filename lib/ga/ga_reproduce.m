function population = ga_reproduce(population, scores, c_prob, m_prob, tournament_size, cmp)
    % tournament selection
    new_generation = ga_tournament_selection(population, scores, 2, @cmp)

    for i = 1:2:length(new_generation)
        % get parents
        parent_1 = new_generation(i);
        parent_2 = new_generation(i + 1);

        % point crossover
        [child_1, child_2] = ga_point_crossover(parent_1, parent_2 c_prob)

        % point mutation
        child_1 = ga_point_mutation(child_1, m_prob);
        child_2 = ga_point_mutation(child_2, m_prob);

        % add back to new generation
        new_generation(:, :, i) = child_1;
        new_generation(:, :, i + 1) = child_2;
    end
end
