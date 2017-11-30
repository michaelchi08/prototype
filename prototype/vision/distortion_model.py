import numpy as np


# def undistort(Idistorted, K)
#     k1 = params.k1
#     k2 = params.k2
#     p1 = params.p1
#     p2 = params.p2
#
#     # Idistorted = single(Idistorted);
#
#     # I = zeros(size(Idistorted))
#     # [i j] = find(~isnan(I))
#
#     # % Xp = the xyz vals of points on the z plane
#     # Xp = inv(K)*[j i ones(length(i),1)]';
#
#     # % Now we calculate how those points distort i.e forward map them through the distortion
#     # r2 = Xp(1,:).^2+Xp(2,:).^2;
#     # x = Xp(1,:);
#     # y = Xp(2,:);
#
#     # x = x.*(1+k1*r2 + k2*r2.^2) + 2*p1.*x.*y + p2*(r2 + 2*x.^2);
#     # y = y.*(1+k1*r2 + k2*r2.^2) + 2*p2.*x.*y + p1*(r2 + 2*y.^2);
#
#     # # u and v are now the distorted cooridnates
#     # u = reshape(fx*x + cx,size(I));
#     # v = reshape(fy*y + cy,size(I));
#
#     # Now we perform a backward mapping in order to undistort the warped image coordinates
#     I = interp2(Idistorted, u, v);


# def radtan(k_1, k_2, k_3, t_1, t_2):
#     """
#
#     Parameters
#     ----------
#     k_1 :
#
#     k_2 :
#
#     k_3 :
#
#     t_1 :
#
#     t_2 :
#
#
#     Returns
#     -------
#
#     """
#     u = x / z
#     v = y / z
#     r = u**2 + v**2
#     d_r = 1 + k_1 * r + k_2 * r**2 + k_3 * r**3
#     d_t = np.array([[2 * u * v * t_1 + (r + 2 * u**2) * t_2]
#                     [2 * u * v * t_2 + (r + 2 * v**2) * t_1]])
#
#     intrinsics = np.array([[f_x, 0.0], [0.0, f_y]])
#     principle = np.array([[p_x], [p_y]])
#     principle + intrinsics * d_r * np.array([[u], [v]]) + d_t
#
#
# def equidistance():
#     """ """
#     pass



