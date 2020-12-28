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
def AlgebricRestrictions(i):
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
    g = -log(-g1) - log(-g2) - log(-g3) - log(-g4) - log(-g5) - log(-g5) - log(-g6) - log(-g7) -log(-g8) - log(-g9) - log(-g10) - log(-g11)

    if i == 1:
        return g
    else:
        return [[g1],
                [g2],
                [g3],
                [g4],
                [g5],
                [g6],
                [g7],
                [g8],
                [g9],
                [g10],
                [g11]]
def Restrictions(i):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')

    inf1 = (x1 - 2.6)**2
    inf2 = (x2 - 0.7)**2
    inf3 = (x3 - 17)**2
    inf4 = (x4 - 7.3)**2
    inf5 = (x5 - 7.8)**2
    inf6 = (x6 - 2.9)**2
    inf7 = (x7 - 5)**2

    sup1 = (3.6 - x1)**2
    sup2 = (0.8 - x2)**2
    sup3 = (28 - x3)**2
    sup4 = (8.3 - x4)**2
    sup5 = (8.3 - x5)**2
    sup6 = (3.9 - x6)**2
    sup7 = (5.5 - x7)**2
    h = (inf1 + sup1)**(1/2) + (inf2 + sup2)**(1/2) + (inf3 + sup3)**(1/2) + (inf4 + sup4)**(1/2) + (inf5 + sup5)**(1/2) + (inf6 + sup6)**(1/2) + (inf7 + sup7)**(1/2)
    g = [[inf1], [inf2], [inf3], [inf4], [inf5], [inf6], [inf7], [sup1], [sup2], [sup3], [sup4], [sup5], [sup6], [sup7]]
    if i == 1:
        return h
    else:
        return g
def differentiation(x):
    f = AlgebricFunction()
    h = AlgebricRestrictions(1)
    g = Restrictions(1)
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')
    g1 = AlgebricRestrictions(0)
    lam_g = lambdify([x1, x2, x3, x4, x5, x6, x7], g1, "numpy")
    restri_g = (lam_g(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))

    lam_h = lambdify([x1, x2, x3, x4, x5, x6, x7], Restrictions(0), "numpy")
    restri_h = (lam_h(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))

    maxG = np.array((restri_g))
    #print(maxG)
    maxH = np.array(restri_h)
    #print(maxH)
    if max(maxG) < 0:
       func = f + h
       print("é menor")
    else:
        func = f + h + g


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
        alphat = alpha * (np.sqrt(abs(1 - m2 ** t)) / (1 - m1 ** t))
        x = x-alphat*pt/(np.sqrt(np.abs(qt))+epsilon)
        dt = alphat*pt/(np.sqrt(np.abs(qt))+epsilon)

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
