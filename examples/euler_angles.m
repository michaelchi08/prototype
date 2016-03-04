function r = deg2rad(degrees)
    r = degrees * pi / 180.0;
endfunction

function d = rad2deg(radians)
    d = radians * 180.0 / pi;
endfunction

function M_body = i2bframe(phi, theta, psi, M_inertial)
    phi = deg2rad(phi);
    theta = deg2rad(theta);
    psi = deg2rad(psi);

    % rotation matrices
    R_1 = [
        1, 0, 0;
        0, cos(phi), sin(phi);
        0, -sin(phi), cos(phi);
    ];

    R_2 = [
        cos(theta), 0, -sin(theta);
        0, 1, 0;
        sin(theta), 0, cos(theta);
    ];

    R_3 = [
        cos(psi), sin(psi), 0;
        -sin(psi), cos(psi), 0;
        0, 0, 1;
    ];

    % rotation matrix
    % el_1 = cos(theta) * cos(psi);
    % el_2 = cos(theta) * sin(psi);
    % el_3 = -sin(theta);
    %
    % el_4 = (sin(phi) * sin(theta) * cos(psi)) - (cos(phi) * sin(psi));
    % el_5 = (sin(phi) * sin(theta) * sin(psi)) + (cos(phi) * cos(psi));
    % el_6 = sin(phi) * cos(theta);
    %
    % el_7 = (cos(phi) * sin(theta) * cos(psi)) + (sin(phi) * sin(psi));
    % el_8 = (cos(phi) * sin(theta) * sin(psi)) - (sin(phi) * cos(psi));
    % el_9 = cos(phi) * cos(theta);
    %
    % rot_mat = [
    %     el_1, el_2, el_3;
    %     el_4, el_5, el_6;
    %     el_7, el_8, el_9;
    % ];

    % result
    M_body = (R_3 * R_2 * R_1) * M_inertial;
endfunction

function M_inertial = b2iframe(phi, theta, psi, M_body)
    phi = deg2rad(phi);
    theta = deg2rad(theta);
    psi = deg2rad(psi);

    % rotation matrices
    R_1 = [
        1.0, 0.0, 0.0;
        0.0, cos(phi), sin(phi);
        0.0, -sin(phi), cos(phi);
    ];

    R_2 = [
        cos(theta), 0.0, -sin(theta);
        0.0, 1.0, 0.0;
        sin(theta), 0.0, cos(theta);
    ];

    R_3 = [
        cos(psi), sin(psi), 0.0;
        -sin(psi), cos(psi), 0.0;
        0.0, 0.0, 1.0;
    ];

    % result
    M_inertial = inverse(R_3 * R_2 * R_1) * M_body;
endfunction

function angrate_body = angrate_i2b(phi, theta, psi, angrate_inertial)
    phi = deg2rad(phi);
    theta = deg2rad(theta);
    psi = deg2rad(psi);

    transformation_matrix = [
        1.0, 0.0, -sin(theta);
        0.0, cos(theta), cos(theta) * sin(theta);
        0.0, -sin(theta), cos(theta) * cos(theta);
    ];

    angrate_body = transformation_matrix * angrate_inertial;
endfunction

function angrate_inertial = angrate_b2i(phi, theta, psi, angrate_body)
    phi = deg2rad(phi);
    theta = deg2rad(theta);
    psi = deg2rad(psi);

    transformation_matrix = [
        1.0, sin(theta) * tan(theta), cos(theta) * tan(theta);
        0.0, cos(theta), -sin(theta);
        0.0, sin(theta) / cos(theta), cos(theta) / cos(theta);
    ];

    angrate_inertial = transformation_matrix * angrate_body;
endfunction

function euler_angle_example()
    % intial values
    P_body = [2.0; 2.5; 0.0];
    roll_inertial = 0.0;
    pitch_inertial = 0.0;
    yaw_inertial = 60.0;

    % convert body frame to inertial frame and back
    P_inertial = b2iframe(
        roll_inertial,
        pitch_inertial,
        yaw_inertial,
        P_body
    );
    P_body_2 = i2bframe(
        roll_inertial,
        pitch_inertial,
        yaw_inertial,
        P_inertial
    );

    % display results
    P_body
    P_inertial
    P_body_2
endfunction

function euler_rate_example()
    angrate_body = [ 10.0; 10.0; 10.0];
    roll_inertial = 10.0;
    pitch_inertial = 20.0;
    yaw_inertial = 30.0;

    angrate_inertial = angrate_b2i(
        roll_inertial,
        pitch_inertial,
        yaw_inertial,
        angrate_body
    )
    angrate_body_2 = angrate_i2b(
        roll_inertial,
        pitch_inertial,
        yaw_inertial,
        angrate_inertial
    )

    % display results
    angrate_body
    angrate_inertial
    angrate_body_2
endfunction

% euler_angle_example()
% euler_rate_example()
