function [yaw, pitch, roll] = quat2euler(q)
    qw = q(1);
    qx = q(2);
    qy = q(3);
    qz = q(4);

    yaw = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx^2 + qy^2));
    pitch = asin(2 * (qw * qy - qz * qx));
    roll = atan2(2 * (qx * qz + qx * qy), 1 - 2 * (qy^2 + qz^2));
endfunction
