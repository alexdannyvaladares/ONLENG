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
def AlgebricRestrictions():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')

    g1 = ((27 / (x1 * x3 * x2 ** 2)) - 1)**2
    g2 = ((397.5 / (x1 * x3 ** 2 * x2 ** 2)) - 1)**2
    g3 = (((1.93 * x4 ** 3) / (x2 * x3 * x6 ** 4)) - 1)**2
    g4 = (((1.93 * x5 ** 3) / (x2 * x3 * x7 ** 4)) - 1)**2
    g5 = (((745 * x4 / (x2 * x3)) ** 2 + 16.9 * 10 ** 6) ** (1 / 2) / (110 * x6 ** 3) - 1)**2
    g6 = ((((745 * x5) / (x2 * x3)) ** 2 + 157.5 * 10 ** 6) ** (1 / 2) / (85 * x7 ** 3) - 1)**2
    g7 = (x2 * x3 / 40 - 1)**2
    g8 = (5 * x2 / x1 - 1)**2
    g9 = (x1 / (12 * x2) - 1)**2
    g10 = ((1.5 * x6 + 1.9) / x4 - 1)**2
    g11 = ((1.1 * x7 + 1.9) / x5 - 1)**2
    g = g1 + g2 + g3 + g4 + g5 + g5 + g6 + g7 + g8 + g9 + g10 + g11
    return g
def Restrictions():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')


    inf1 = (x1 - 2.6) ** 2
    inf2 = (x2 - 0.7) ** 2
    inf3 = (x3 - 17) ** 2
    inf4 = (x4 - 7.3) ** 2
    inf5 = (x5 - 7.8) ** 2
    inf6 = (x6 - 2.9) ** 2
    inf7 = (x7 - 5) ** 2

    sup1 = (3.6 - x1) ** 2
    sup2 = (0.8 - x2) ** 2
    sup3 = (28 - x3) ** 2
    sup4 = (8.3 - x4) ** 2
    sup5 = (8.3 - x5) ** 2
    sup6 = (3.9 - x6) ** 2
    sup7 = (5.5 - x7) ** 2
    h = inf1 + inf2 + inf3 + inf4 + inf5 + inf6 + inf7 + sup1 + sup2 + sup3 + sup4 + sup5 + sup6 + sup7

    return h
def differentiation(x):
    f = AlgebricFunction()
    h = AlgebricRestrictions()
    g = Restrictions()
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')
    g1 = AlgebricRestrictions()
    lam_g = lambdify([x1, x2, x3, x4, x5, x6, x7], g1, "numpy")
    restri = (lam_g(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
    #if restri < 0:
     #   func = f + h

    #else:
    func = f + h + g
        #print("restrições mal")

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
def differentiation2():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')
    grad = np.zeros([7,1])
    f = AlgebricFunction()
    h = AlgebricRestrictions()
    g = Restrictions()
    func = f + h + g
    gradx1 = func.diff(x1)
    gradx2 = func.diff(x2)
    gradx3 = func.diff(x3)
    gradx4 = func.diff(x4)
    gradx5 = func.diff(x5)
    gradx6 = func.diff(x6)
    gradx7 = func.diff(x7)

    dx1dx1 = gradx1.diff(x1)
    dx1dx2 = gradx1.diff(x2)
    dx1dx3 = gradx1.diff(x3)
    dx1dx4 = gradx1.diff(x4)
    dx1dx5 = gradx1.diff(x5)
    dx1dx6 = gradx1.diff(x6)
    dx1dx7 = gradx1.diff(x7)

    dx2dx1 = gradx2.diff(x1)
    dx2dx2 = gradx2.diff(x2)
    dx2dx3 = gradx2.diff(x3)
    dx2dx4 = gradx2.diff(x4)
    dx2dx5 = gradx2.diff(x5)
    dx2dx6 = gradx2.diff(x6)
    dx2dx7 = gradx2.diff(x7)

    dx3dx1 = gradx3.diff(x1)
    dx3dx2 = gradx3.diff(x2)
    dx3dx3 = gradx3.diff(x3)
    dx3dx4 = gradx3.diff(x4)
    dx3dx5 = gradx3.diff(x5)
    dx3dx6 = gradx3.diff(x6)
    dx3dx7 = gradx3.diff(x7)

    dx4dx1 = gradx4.diff(x1)
    dx4dx2 = gradx4.diff(x2)
    dx4dx3 = gradx4.diff(x3)
    dx4dx4 = gradx4.diff(x4)
    dx4dx5 = gradx4.diff(x5)
    dx4dx6 = gradx4.diff(x6)
    dx4dx7 = gradx4.diff(x7)

    dx5dx1 = gradx5.diff(x1)
    dx5dx2 = gradx5.diff(x2)
    dx5dx3 = gradx5.diff(x3)
    dx5dx4 = gradx5.diff(x4)
    dx5dx5 = gradx5.diff(x5)
    dx5dx6 = gradx5.diff(x6)
    dx5dx7 = gradx5.diff(x7)

    dx6dx1 = gradx6.diff(x1)
    dx6dx2 = gradx6.diff(x2)
    dx6dx3 = gradx6.diff(x3)
    dx6dx4 = gradx6.diff(x4)
    dx6dx5 = gradx6.diff(x5)
    dx6dx6 = gradx6.diff(x6)
    dx6dx7 = gradx6.diff(x7)

    dx7dx1 = gradx7.diff(x1)
    dx7dx2 = gradx7.diff(x2)
    dx7dx3 = gradx7.diff(x3)
    dx7dx4 = gradx7.diff(x4)
    dx7dx5 = gradx7.diff(x5)
    dx7dx6 = gradx7.diff(x6)
    dx7dx7 = gradx7.diff(x7)

    doubleGrad = [[dx1dx1, dx1dx2, dx1dx3, dx1dx4, dx1dx5, dx1dx6, dx1dx7],
                  [dx2dx1, dx2dx2, dx2dx3, dx2dx4, dx2dx5, dx2dx6, dx2dx7],
                  [dx3dx1, dx3dx2, dx3dx3, dx3dx4, dx3dx5, dx3dx6, dx3dx7],
                  [dx4dx1, dx4dx2, dx4dx3, dx4dx4, dx4dx5, dx4dx6, dx4dx7],
                  [dx5dx1, dx5dx2, dx5dx3, dx5dx4, dx5dx5, dx5dx6, dx5dx7],
                  [dx6dx1, dx6dx2, dx6dx3, dx6dx4, dx6dx5, dx6dx6, dx6dx7],
                  [dx7dx1, dx7dx2, dx7dx3, dx7dx4, dx7dx5, dx7dx6, dx7dx7]]


    return doubleGrad
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
        #x[3] = math.ceil(x[3])
        j = j+1
        gamma = (x-xn)*(dx-dxn)/np.linalg.norm((dx-dxn)**2)
        print(func(x))
        #if (func(x)/func(xn)>0.9999) and j > 10:
         #   print(func(x))
          #  print(x)
           # break

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
