function retval = wrapTo360(euler_angle)
    retval = mod(euler_angle, 360);
endfunction
