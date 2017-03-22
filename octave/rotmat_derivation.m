#!/usr/bin/octave
% pkg install -forge symbolic
pkg load symbolic

syms alpha beta gamma

Rx = [[1.0, 0.0, 0.0];
      [0.0, cos(alpha), sin(alpha)];
      [0.0, -sin(alpha), cos(alpha)]];

Ry = [[cos(beta), 0.0, -sin(beta)];
      [0.0, 1.0, 0.0];
      [sin(beta), 0.0, cos(beta)]];

Rz = [[cos(gamma), sin(gamma), 0.0];
      [-sin(gamma), cos(gamma), 0.0];
      [0.0, 0.0, 1.0]];

% euler 3-2-1  (inertial to body)
R321 = Rz * Ry * Rx


% % euler 1-2-3  (body to inertial)
R123 = Rx * Ry * Rz
