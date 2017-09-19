alpha = 0.1
beta = 0.2
gamma = 0.3

Rx = [[1.0, 0.0, 0.0];
      [0.0, cos(alpha), -sin(alpha)];
      [0.0, sin(alpha), cos(alpha)]];
Ry = [[cos(beta), 0.0, sin(beta)];
      [0.0, 1.0, 0.0];
      [-sin(beta), 0.0, cos(beta)]];
Rz = [[cos(gamma), -sin(gamma), 0.0];
      [sin(gamma), cos(gamma), 0.0];
      [0.0, 0.0, 1.0]];

v = [1;2;3]
R = Rx * Ry * Rz
R * v


% Rx = [[1.0, 0.0, 0.0];
%       [0.0, cos(alpha), sin(alpha)];
%       [0.0, -sin(alpha), cos(alpha)]];
% Ry = [[cos(beta), 0.0, -sin(beta)];
%       [0.0, 1.0, 0.0];
%       [sin(beta), 0.0, cos(beta)]];
% Rz = [[cos(gamma), sin(gamma), 0.0];
%       [-sin(gamma), cos(gamma), 0.0];
%       [0.0, 0.0, 1.0]];
%
% R = Rx * Ry * Rz;

% Rz * [1; 0; 0] + [-1; 0; 0]
