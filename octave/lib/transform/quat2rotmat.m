function R = quat2rotmat(qw, qx, qy, qz)
    % inhomogeneous form
    R = [
        % row 1
        1 - 2 * qy^2 - 2 * qz^2,
        2 * qx * qy - 2 * qz * qw,
        2 * qx * qz + 2 * qy * qw;
        % row 2
        2 * qx * qy + 2 * qz * qw,
        1 - 2 * qx^2 - 2 * qz^2,
        2 * qy * qz - 2 * qx * qw;
        % row 3
        2 * qx * qz - 2 * qy * qw,
        2 * qy * qz - 2 * qx * qw,
        1 - 2 * qx^2 - 2 * qy^2;
    ];

    % % homogeneous form
    % R = [
    %     % row 1
    %     qx^2 + qx^2 - qy^2 - qz^2,
    %     2 * (qx * qy - qw * qz),
    %     2 * (qw * qy + qx * qz);
    %     % row 2
    %     2 * (qx * qy + qw * qz),
    %     qw^2 - qx^2 + qy^2 - qz^2,
    %     2 * (qy * qz - qw * qx);
    %     % row 3
    %     2 * (qx * qz - qw * qy),
    %     2 * (qw * qx + qy * qz),
    %     qw^2 - qx^2 - qy^2 + qz^2;
    % ];
endfunction
