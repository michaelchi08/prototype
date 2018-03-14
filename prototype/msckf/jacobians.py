import sympy


def measurement_model():
    X, Y, Z = sympy.symbols("X Y Z")
    # C_CiG = sympy.symbols("C_CiG")

    # p_G_fj = sympy.symbols("p_G_fj")
    X_G_fj = sympy.symbols("X_G_fj")
    Y_G_fj = sympy.symbols("Y_G_fj")
    Z_G_fj = sympy.symbols("Z_G_fj")

    # p_G_Ci = sympy.symbols("p_G_Ci")
    X_G_Ci = sympy.symbols("X_G_Ci")
    Y_G_Ci = sympy.symbols("Y_G_Ci")
    Z_G_Ci = sympy.symbols("Z_G_Ci")

    R11, R12, R13 = sympy.symbols("R11, R12, R13")
    R21, R22, R23 = sympy.symbols("R21, R22, R23")
    R31, R32, R33 = sympy.symbols("R31, R32, R33")
    C_CiG = sympy.Matrix([[R11, R12, R13],
                          [R21, R22, R23],
                          [R31, R32, R33]])
    p_G_fj = sympy.Matrix([X_G_fj, Y_G_fj, Z_G_fj])
    p_G_Ci = sympy.Matrix([X_G_Ci, Y_G_Ci, Z_G_Ci])

    f = C_CiG.dot(p_G_fj - p_G_Ci)
    func = sympy.Matrix(f)
    print(func.jacobian([X_G_Ci]))
    print(func.jacobian([Y_G_Ci]))

    # print(f)
    # print(f.jacobian(X_G_Ci))


    # f = 1 / Z * sympy.Matrix([X, Y])
    # print(f)
