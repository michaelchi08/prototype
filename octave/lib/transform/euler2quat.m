function q = euler2quat(alpha, beta, gamma, euler_seq)
    c1 = cos(alpha / 2.0);
    c2 = cos(beta / 2.0);
    c3 = cos(gamma / 2.0);
    s1 = sin(alpha / 2.0);
    s2 = sin(beta / 2.0);
    s3 = sin(gamma / 2.0);

    switch (euler_seq)
    case "123"
        % euler 1-2-3 to quaternion
        w = c1 * c2 * c3 - s1 * s2 * s3;
        x = s1 * c2 * c3 + c1 * s2 * s3;
        y = c1 * s2 * c3 - s1 * c2 * s3;
        z = c1 * c2 * s3 + s1 * s2 * c3;

    case "321"
        % euler 3-2-1 to quaternion
        w = c1 * c2 * c3 + s1 * s2 * s3;
        x = s1 * c2 * c3 - c1 * s2 * s3;
        y = c1 * s2 * c3 + s1 * c2 * s3;
        z = c1 * c2 * s3 - s1 * s2 * c3;

    otherwise
        printf("Error! Invalid euler sequence [%s]\n", euler_seq);
    endswitch

    q = [w; x; y; z];
endfunction
