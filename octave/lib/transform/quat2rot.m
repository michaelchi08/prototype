function R = quat2rot(qw, qx, qy, qz)
    % inhomogeneous form
    R11 = 1 - 2 * qy^2 - 2 * qz^2;
    R12 = 2 * qx * qy + 2 * qz * qw;
    R13 = 2 * qx * qz - 2 * qy * qw;

    R21 = 2 * qx * qy - 2 * qz * qw;
    R22 = 1 - 2 * qx^2 - 2 * qz^2;
    R23 = 2 * qy * qz + 2 * qx * qw;

    R31 = 2 * qx * qz + 2 * qy * qw;
    R32 = 2 * qy * qz - 2 * qx * qw;
    R33 = 1 - 2 * qx^2 - 2 * qy^2;

    % % homogeneous form
    % R11 = qx^2 + qx^2 - qy^2 - qz^2;
    % R12 = 2 * (qx * qy - qw * qz);
    % R13 = 2 * (qw * qy + qx * qz);
    %
    % R21 = 2 * (qx * qy + qw * qz);
    % R22 = qw^2 - qx^2 + qy^2 - qz^2;
    % R23 = 2 * (qy * qz - qw * qx);
    %
    % R31 = 2 * (qx * qz - qw * qy);
    % R32 = 2 * (qw * qx + qy * qz);
    % R33 = qw^2 - qx^2 - qy^2 + qz^2;

    R = [R11, R12, R13;
         R21, R22, R23;
         R31, R32, R33];
endfunction
