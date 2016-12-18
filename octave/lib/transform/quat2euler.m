function [psi, theta, phi] = quat2euler(qw, qx, qy, qz, euler_seq)
    switch (euler_seq)
    case "123"
        % euler 1-2-3
        psi = atan2(2 * (qx * qw - qy * qz), (qw^2 - qx^2 - qy^2 + qz^2));
        theta = asin(2 * (qx * qz+qy * qw));
        phi = atan2(2 * (qz * qw - qx * qy), (qw^2 + qx^2 - qy^2 - qz^2));

    case "321"
        % euler 3-2-1
        psi = atan2(2 * (qx * qy + qz * qw), (qw^2 + qx^2 - qy^2 - qz^2));
        theta = asin(2 * (qy * qw - qx * qz));
        phi = atan2(2 * (qx * qw + qz * qy), (qw^2 - qx^2 - qy^2 + qz^2));

    otherwise
        printf("Error! Invalid euler sequence [%s]\n", euler_seq);
    endswitch
endfunction
