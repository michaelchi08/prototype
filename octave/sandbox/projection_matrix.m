#!/usr/bin/octave -qf
addpath('./lib/transform');
addpath('./lib/util');

graphics_toolkit("gnuplot")


C_w = [0; 0; 0.0]

w_R_c = euler123(deg2rad(0), deg2rad(0), deg2rad(0))

X_w1 = [5.8; 5.8; 10; 1]
X_w2 = [5.8; -5.8; 10; 1]
X_w3 = [-5.8; -5.8; 10; 1]
X_w4 = [-5.8; 5.8; 10; 1]


% X_c = w_R_c * (X_w - C_w)

T = [w_R_c, -w_R_c * C_w;
     0, 0, 0, 1]

K = [554.26, 0.0, 320;
    0.0, 554.26, 320;
    0.0, 0.0, 1.0;];

% P = K * [w_R_c', -w_R_c' * C_w];
P = K * [w_R_c, C_w]

X_c1 = P * X_w1;
X_c1(1) = X_c1(1) / X_c1(3);
X_c1(2) = X_c1(2) / X_c1(3);
X_c1(3) = X_c1(3) / X_c1(3);

X_c2 = P * X_w2;
X_c2(1) = X_c2(1) / X_c2(3);
X_c2(2) = X_c2(2) / X_c2(3);
X_c2(3) = X_c2(3) / X_c2(3);

X_c3 = P * X_w3;
X_c3(1) = X_c3(1) / X_c3(3);
X_c3(2) = X_c3(2) / X_c3(3);
X_c3(3) = X_c3(3) / X_c3(3);

X_c4 = P * X_w4;
X_c4(1) = X_c4(1) / X_c4(3);
X_c4(2) = X_c4(2) / X_c4(3);
X_c4(3) = X_c4(3) / X_c4(3);

function fx = focal(fov, image_width)
    fx = (image_width / 2.0) / tan(deg2rad(fov) / 2.0);
endfunction

rad2deg(atan(6 / 10))

% plot
figure;
hold on;

X_c1
X_c2
X_c3
X_c4

plot3([0; X_w1(1)], [0; X_w1(2)], [0; X_w1(3)]);
plot3([0; X_w2(1)], [0; X_w2(2)], [0; X_w2(3)]);
plot3([0; X_w3(1)], [0; X_w3(2)], [0; X_w3(3)]);
plot3([0; X_w4(1)], [0; X_w4(2)], [0; X_w4(3)]);

axis equal;
view(3);
pause;
