from numpy import random
import sympy
from sympy import *
from sympy.utilities.lambdify import lambdify, implemented_function
import numpy as np
import math
def numberPick():
    switch = {
        0: [2.6, 3.6],
        1: [0.7, 0.8],
        2: [17, 28],
        3: [7.3, 8.3],
        4: [7.8, 8.3],
        5: [2.9, 3.9],
        6: [5, 5.5]
    }
    x = random.rand(7,1)
    for i in range(7):
        m = switch.get(i)
        x[i] = x[i]*(m[1]-m[0]) + m[0]
    x[0] = 2
    print(x)

    return x
'''
def function():
    f = 0.7854 * x[0] * x[1] ** 2 * \
        (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934) \
        - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2) \
        + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)

    return f'''
def AlgebricFunction():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')
    f = 0.7854 * x1 * x2 ** 2 * \
        (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934) \
        - 1.508 * x1 * (x6 ** 2 + x7 ** 2) \
        + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2)

    return f
def AlgebricRestrictions(i, x):
    value = np.zeros([11,1])
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')


    g1 = ((27 / (x1 * x3 * x2 ** 2)) - 1)
    g2 = ((397.5 / (x1 * x3 ** 2 * x2 ** 2)) - 1)
    g3 = (((1.93 * x4 ** 3) / (x2 * x3 * x6 ** 4)) - 1)
    g4 = (((1.93 * x5 ** 3) / (x2 * x3 * x7 ** 4)) - 1)
    g5 = (((745 * x4 / (x2 * x3)) ** 2 + 16.9 * 10 ** 6) ** (1 / 2) / (110 * x6 ** 3) - 1)
    g6 = ((((745 * x5) / (x2 * x3)) ** 2 + 157.5 * 10 ** 6) ** (1 / 2) / (85 * x7 ** 3) - 1)
    g7 = (x2 * x3 / 40 - 1)
    g8 = (5 * x2 / x1 - 1)
    g9 = (x1 / (12 * x2) - 1)
    g10 = ((1.5 * x6 + 1.9) / x4 - 1)
    g11 = ((1.1 * x7 + 1.9) / x5 - 1)

    value_g1 = lambdify([x1, x2, x3, x4, x5, x6, x7], g1, "numpy")
    value_g2 = lambdify([x1, x2, x3, x4, x5, x6, x7], g2, "numpy")
    value_g3 = lambdify([x1, x2, x3, x4, x5, x6, x7], g3, "numpy")
    value_g4 = lambdify([x1, x2, x3, x4, x5, x6, x7], g4, "numpy")
    value_g5 = lambdify([x1, x2, x3, x4, x5, x6, x7], g5, "numpy")
    value_g6 = lambdify([x1, x2, x3, x4, x5, x6, x7], g6, "numpy")
    value_g7 = lambdify([x1, x2, x3, x4, x5, x6, x7], g7, "numpy")
    value_g8 = lambdify([x1, x2, x3, x4, x5, x6, x7], g8, "numpy")
    value_g9 = lambdify([x1, x2, x3, x4, x5, x6, x7], g9, "numpy")
    value_g10 = lambdify([x1, x2, x3, x4, x5, x6, x7], g10, "numpy")
    value_g11 = lambdify([x1, x2, x3, x4, x5, x6, x7], g11, "numpy")

    value[0] = (value_g1(x[0], x[1], x[2], x[3], x[4], x[5], x[6])) > 0
    value[1] = (value_g2(x[0], x[1], x[2], x[3], x[4], x[5], x[6])) > 0
    value[2] = (value_g3(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0
    value[3] = (value_g4(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0
    value[4] = (value_g5(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0
    value[5] = (value_g6(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0
    value[6] = (value_g7(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0
    value[7] = (value_g8(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0
    value[8] = (value_g9(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0
    value[9] = (value_g10(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0
    value[10] = (value_g11(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))> 0


    g = -log(-g1) - log(-g2) - log(-g3) - log(-g4) - log(-g5) - log(-g5) - log(-g6) - log(-g7) -log(-g8) - log(-g9) - log(-g10) - log(-g11)

    if i == 1:
        return g
    else:
        return value
def Restrictions(i, x):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')

    inf1 = lambdify(x1, (x1 - 2.6), "numpy")
    inf2 = lambdify(x2,(x2 - 0.7), "numpy")
    inf3 = lambdify(x3,(x3 - 17), "numpy")
    inf4 = lambdify(x4,(x4 - 7.3), "numpy")
    inf5 = lambdify(x5,(x5 - 7.8), "numpy")
    inf6 = lambdify(x6,(x6 - 2.9), "numpy")
    inf7 = lambdify(x7,(x7 - 5), "numpy")

    sup1 = lambdify(x1, (3.6 - x1), "numpy")
    sup2 = lambdify(x2, (0.8 - x2), "numpy")
    sup3 = lambdify(x3, (28 - x3), "numpy")
    sup4 = lambdify(x4, (8.3 - x4), "numpy")
    sup5 = lambdify(x5, (8.3 - x5), "numpy")
    sup6 = lambdify(x6, (3.9 - x6), "numpy")
    sup7 = lambdify(x7, (5.5 - x7), "numpy")

    g = (np.array([inf1(x[0]), inf2(x[1]), inf3(x[2]), inf4(x[3]), inf5(x[4]), inf6(x[5]), inf7(x[6])])<0)*1
    h = (np.array([sup1(x[0]), sup2(x[1]), sup3(x[2]), sup4(x[3]), sup5(x[4]), sup6(x[5]), sup7(x[6])])<0)*1


    return g, h
def differentiation(x):
    f = AlgebricFunction()
    #h = AlgebricRestrictions(1,x)
    [hinf, hsup] = Restrictions(1, x)
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')
    g = AlgebricRestrictions(0,x)
    print(hinf)
    g = g*10**22
    hinf = hinf*(10**22)
    hsup = -hsup*10**22
    func = f


    dx1 = lambdify([x1, x2, x3, x4, x5, x6, x7], func.diff(x1), 'numpy')
    dx2 = lambdify([x1, x2, x3, x4, x5, x6, x7], func.diff(x2), 'numpy')
    dx3 = lambdify([x1, x2, x3, x4, x5, x6, x7], func.diff(x3), 'numpy')
    dx4 = lambdify([x1, x2, x3, x4, x5, x6, x7], func.diff(x4), 'numpy')
    dx5 = lambdify([x1, x2, x3, x4, x5, x6, x7], func.diff(x5), 'numpy')
    dx6 = lambdify([x1, x2, x3, x4, x5, x6, x7], func.diff(x6), 'numpy')
    dx7 = lambdify([x1, x2, x3, x4, x5, x6, x7], func.diff(x7), 'numpy')
    grad = np.array([dx1(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dx2(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dx3(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dx4(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dx5(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dx6(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dx7(x[0], x[1], x[2], x[3], x[4], x[5], x[6])])
    grad = grad + g + hinf + hsup

    return grad

def symbolToNumeric(grad,x):
    doubleGrad = np.zeros([7,7])
    for i in range(7):
        for j in range(7):

            doubleGrad[i,j] = lambdify([x[0], x[1], x[2], x[3], x[4], x[5], x[6]], grad[i][j], 'numpy')



    return doubleGrad
def optimal(grad, x, gamma, maxIter, xn):
    j = 0
    delta = 0
    xValue = np.zeros([7,1])
    valueFunc = 1000
    p = 0
    q = 0
    m1 = 0.9
    m2 = 0.999
    t=1
    dx = 0
    dt = 0
    alpha = 0.001
    epsilon = 1e-6
    while j < maxIter:
        #symbolToNumeric(doubleGrad,x)
        t = t+dt
        dxn = dx
        xn = x
        dx = np.squeeze(grad(x))
        p = m1*p+(1-m1)*dx
        q = m2*q+(1-m2)*dx**2
        pt = p/(1-m1**t)
        qt = q/(1-m2**t)
        alphat = alpha * ((abs(1 - m2 ** t)) / (1 - m1 ** t))**(1/2)
        x = x-alphat*pt/((np.abs(qt))+epsilon)**(1/2)
        dt = alphat*pt/((np.abs(qt))+epsilon)**(1/2)
        print(func(x))
        #delta = 0.01*gamma*dx + 0.9*delta
        #x = x - delta
        #x[3] = math.floor(x[3])
        j = j+1
        #gamma = (x-xn)*(dx-dxn)/np.linalg.norm((dx-dxn)**2)


        if ((func(x)/func(xn)>0.99999) and func(x)/func(xn)<1.00001) and j > 10:
            #print(func(x))
            #print(x)
            break

    return x
def func(x):
    y = 0.7854 * x[0] * (x[1] ** 2) * (3.3333 * (x[2] ** 2)+ 14.9334 * x[2] - 43.0934) - 1.508 * x[0] * ((x[5] ** 2) + (x[6] ** 2)) +7.4777*(x[5]**3 + x[6]**3)+ 0.7854 * (x[3] * (x[5] ** 2) + x[4] * (x[6] ** 2))
    return y

x = np.squeeze(numberPick())
iterations = 10e5
xn = np.squeeze(np.zeros([7, 1]))
gamma = np.squeeze(np.ones([7,1]))*0.0001
#doubleGrad = differentiation2()
x = optimal(differentiation, x, gamma, iterations, xn)
y = func(x)
print(x)
print(y)
