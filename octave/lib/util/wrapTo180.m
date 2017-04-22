function retval = wrapTo180(euler_angle)
    retval = mod((euler_angle + 180.0), 360) - 180.0;
endfunction
