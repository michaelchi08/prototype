function Ry = rotmat_y(theta)
    Ry = [cos(theta), 0.0, -sin(theta);
          0.0, 1.0, 0.0;
          sin(theta), 0.0, cos(theta)];
endfunction
