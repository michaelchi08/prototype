function q = euler2quat(yaw, pitch, roll)
    c1 = cos(roll / 2.0);
    c2 = cos(pitch / 2.0);
    c3 = cos(yaw / 2.0);
    s1 = sin(roll / 2.0);
    s2 = sin(pitch / 2.0);
    s3 = sin(yaw / 2.0);

    w = c1 * c2 * c3 - s1 * s2 * s3;
    x = -c1 * s2 * s3 + c2 * c3 * s1;
    y = c1 * c3 * s2 + s1 * c2 * s3;
    z = c1 * c2 * s3 - s1 * c3 * s2;

    q = [w; x; y; z];
endfunction
