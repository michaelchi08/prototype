function Rx = rotmat_x(theta)
    Rx = [1.0, 0.0, 0.0;
          0.0, cos(theta), sin(theta);
          0.0, -sin(theta), cos(theta)];
endfunction
