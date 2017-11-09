# def bresenham2d(x1, y1, x2, y2):
#     """
#     Bresenham's Algorithm
#
#     Args:
#
#         (x1,y1): Start position
#         (x2,y2): End position
#
#     Returns:
#         x y: the line coordinates from (x1,y1) to (x2,y2)
#     """
#     x1=round(x1)
#     x2=round(x2)
#     y1=round(y1)
#     y2=round(y2)
#
#     dx=abs(x2-x1)
#     dy=abs(y2-y1)
#     steep = abs(dy) > abs(dx)
#
#     if steep:
#         t=dx
#         dx=dy
#         dy=t
#
#     if dy == 0:
#         q = zeros(dx+1, 1)
#     else:
#         q=[0;diff(mod([floor(dx/2):-dy:-dy*dx+floor(dx/2)]',dx))>=0];
#
#
#     if steep:
#         if y1<=y2:
#             y=[y1:y2]'
#         else:
#             y=[y1:-1:y2]'
#
#         if x1<=x2:
#             x=x1+cumsum(q)
#         else:
#             x=x1-cumsum(q')
#     else:
#         if x1<=x2 x=[x1:x2]'; else x=[x1:-1:x2]'
#         if y1<=y2 y=y1+cumsum(q);else y=y1-cumsum(q'
#
