function parents = ga_tournament_selection(population, scores, tournament_size, cmp)
    parents = zeros(1, length(population(:, :, 1)), length(population));

    % tournament selection
    for i = 1:length(population)
        % pre-load best with random sample
        rand_index = floor(rand * length(population));
        if (rand_index == 0)
            rand_index = 1;
        end
        best = population(rand_index);
        best_score = scores(rand_index);

        % random select individuals to fill tournament
        for j = 1:tournament_size
            % random select contender
            rand_index = floor(rand * length(population));
            if (rand_index == 0)
                rand_index = 1;
            end
            contender = population(rand_index);
            contender_score = scores(rand_index);

            % compare to see if contender is better than best
            if (cmp(contender_score, best_score))
                best = contender;
                best_score = contender_score;
            end
        end

        % add best into parents
        parents(:, :, i) = best;
    end
end
