# Random Sample Consensus (RANSAC)

The Random Sample Consensus algorithm propsed by Fischler and Bolles is a
general parameter estimation approach designed to cope with a alarge proportion
of outliers in the input data. It is an algorithm that was developed within the
Computer Vision community.

RANSAC is a resampling technique that generates candidate solutiosn by using
the minimum number of observations (data points) requried to estimate the
underlying model parameters. As pointed out by Fischler and Bolls, unlike
conventional smapling techniques that use as much of the data as possible to
obtain an initial solution and then proceed to prune outliers, RANSAC uses the
smallest set possible and proceeds to enalarge this set with consistent data
points.

The basic algorithm is summaries as follows:


    1. Select a random subset of the original data. Call this subset the
       hypothetical inliers.

    2. A model is fitted to the set of hypothetical inliers.

    3. All other data are then tested against the fitted model. Those
       points that fit the estimated model well, according to some
       model-specific loss function, are considered as part of the
       consensus set.

    4. The estimated model is reasonably good if sufficiently many
       points have been classified as part of the consensus set.

    5. Afterwards, the model may be improved by reestimating it
       using all members of the consensus set.


The number of iternations $N$ is chosen high enough to ensure that the
probability $p$ (usually set ot 0.99) that at least one f the sets of random
samples does not include an outlier. Let $u$ represent the probability that any
selected data point is an inlier and $v = 1 - u$ the probability of observing
an outlier. $N$ iteramtions of the minimum number of points denoted $m$ are
required, where:

\begin{equation}
    1 - p = (1 - u^{m})^{N}
\end{equation}

and thus with some maniulation,

\begin{equation}
    N = \dfrac{log(1 - p)}{log(1 - (1 - v)^{m})}
\end{equation}


    #!/usr/bin/octave

    function [param1, param2] = ransac_demo(data,num,iter,threshDist,inlierRatio)
    % data: a 2xn dataset with #n data points
    % num: the minimum number of points. For line fitting problem, num=2
    % iter: the number of iterations
    % threshDist: threshold of the distances between points and fitting line
    % inlierRatio: the threshold of the number of inliers

    % Plot the data points
    figure;
    plot(data(1,:),
    data(2,:),'o');
    hold on;

    number = size(data,2); % Total number of points
    bestInNum = 0; % Best fitting line with largest number of inliers
    param1=0; % parameters for best fitting line
    param2=0;

    for i=1:iter
        % Randomly select 2 points
        idx = randperm(number,num); sample = data(:,idx);

        % Compute the distances between all points with the fitting line
        kLine = sample(:,2)-sample(:,1);% two points relative distance
        kLineNorm = kLine/norm(kLine);

        % Ax+By+C=0 A=-kLineNorm(2),B=kLineNorm(1)
        normVector = [-kLineNorm(2),kLineNorm(1)];
        distance = normVector*(data - repmat(sample(:,1),1,number));

        % Compute the inliers with distances smaller than the threshold
        inlierIdx = find(abs(distance)<=threshDist);
        inlierNum = length(inlierIdx);

        % Update number of inliers and fitting model if better model found
        if inlierNum>=round(inlierRatio*number) && inlierNum>bestInNum
            bestInNum = inlierNum;
            parameter1 = (sample(2,2)-sample(2,1))/(sample(1,2)-sample(1,1));
            parameter2 = sample(2,1)-parameter1*sample(1,1);
            param1=parameter1; param2=parameter2;
        end
    end

    % Plot the best fitting line
    xAxis = -number/2:number/2;
    yAxis = param1*xAxis + param2;
    plot(xAxis,yAxis,'r-','LineWidth',2);
