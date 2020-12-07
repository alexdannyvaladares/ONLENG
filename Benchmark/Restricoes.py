import numpy as np
import time
from sympy import *

learningRate = 0.001
iterations = 1500
N = 10
x1 = np.linspace(2.6, 3.6, N) #largura da cremalheira
x2 = np.linspace(0.7, 0.8, N) #módulo da engrenagem
x3 = np.arange(17,29) #nº de dentes do pinhão
x4 = np.linspace(7.3, 8.3, N) #comprimento do primeiro veio entre rolamentos
x5 = np.linspace(7.8, 8.3, N) #comprimento do segundo veio entre rolamentos
x6 = np.linspace(2.9, 3.9, N) #diâmetro do primeiro veio
x7 = np.linspace(5, 5.5, N) #diâmetro do segundo veio

x = np.array([[1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1]])


restri = np.zeros(11)
newValue = 0
oldValue = 10**5
#Restrições
'''
g1=(27/(x1[j]*x3[i]*x2[l]**2))-1
g2=((397.5)/x1[j]*x3[i]*x2[l]**2)-1
g3=((1.93*x4[m]**3)/x2[l]*x3[i]*x6[o]**2)-1
g4=((1.93*x5[n]**3)/x2[l]*x3[i]*x7[p]**4)-1
g5=(np.sqrt((745*x4[m]/x2[l]*x3[i])**2 + 16.9*10**6)/110*x6[o]**3)-1
g6 = np.sqrt(((745 * x5[n]) / (x2[l] * x3[i])) ** 2 + 157.5 * 10 ** 6) / (85 * x7[p] ** 3) - 1
g7 = x2[l] * x3[i] / 40 - 1
g8 = 5 * x2[l] / x1[j] - 1
g9 = x1[j] / (12 * x2[l]) - 1
g10 = (1.5 * x6[o] + 1.9) / x4[m] - 1
g11 = (1.1 * x7[p] + 1.9) / x5[n] - 1
restri = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]

if max(restri) <= 0:
    f = 0.7854 * x1[j] * x2[l] ** 2 * \
        (3.3333 * x3[i] ** 2 + 14.9934 * x3[i] - 43.0934) \
        - 1.508 * x1[j] * (x6[o] ** 2 + x7 ** 2) \
        + 0.7854 * (x4[m] * x6[o] ** 2 + x5[n] * x7[p] ** 2)

    if oldValue > f:
        oldValue = f
'''
def gradient(x):
    x = np.squeeze(np.transpose(x))
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')
    y = 0.7854 * x1 * x2 ** 2 * \
        (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934) \
        - 1.508 * x1 * (x6 ** 2 + x7 ** 2) \
        + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2)
    dx1 = lambdify([x1, x2, x3, x4, x5, x6, x7], y.diff(x1), 'numpy')
    dx2 = lambdify([x1, x2, x3, x4, x5, x6, x7], y.diff(x2), 'numpy')
    dx3 = lambdify([x1, x2, x3, x4, x5, x6, x7], y.diff(x3), 'numpy')
    dx4 = lambdify([x1, x2, x3, x4, x5, x6, x7], y.diff(x4), 'numpy')
    dx5 = lambdify([x1, x2, x3, x4, x5, x6, x7], y.diff(x5), 'numpy')
    dx6 = lambdify([x1, x2, x3, x4, x5, x6, x7], y.diff(x6), 'numpy')
    dx7 = lambdify([x1, x2, x3, x4, x5, x6, x7], y.diff(x7), 'numpy')

    grad = np.array([dx1(x[0],x[1],x[2],x[3],x[4],x[5],x[6]),
                     dx2(x[0],x[1],x[2],x[3],x[4],x[5],x[6]),
                     dx3(x[0],x[1],x[2],x[3],x[4],x[5],x[6]),
                     dx4(x[0],x[1],x[2],x[3],x[4],x[5],x[6]),
                     dx5(x[0],x[1],x[2],x[3],x[4],x[5],x[6]),
                     dx6(x[0],x[1],x[2],x[3],x[4],x[5],x[6]),
                     dx7(x[0],x[1],x[2],x[3],x[4],x[5],x[6])])
    '''
    dx1 = x[0]**2*(2.61797*x[2]**2+11.7287*x[2]-33.8456)-1.508*x[5]**2-1.508*x[6]**2
    dx2 = 5.23595*x[0]*x[1]*(x[2]**2+4.48006*x[2]-12.9281)
    dx3 = 5.23595*x[0]*x[1]**2*(x[2]+2.24003)
    dx4 = 0.7854*x[5]**2
    dx5 = 0.7854*x[6]**2
    dx6 = x[5]*(-3.016*x[0] + 1.5708*x[3]+22.4331*x[5])
    dx7 = x[6]*(-3.016*x[0]+1.5708*x[4]+22.4331*x[6])
    

    
    '''
    return grad

def gradientDescent(x, iterations, learningRate):
    xn = np.zeros(len(x))
    dxn = np.zeros([len(x), 1])
    gamma = np.ones([len(x), 1]) * 0.0001

    dx = gradient(x)*learningRate + np.squeeze(penal(x))
    #print(np.sum(dx))



    for i in range(len(x)):
        xn[i] = x[i] - gamma[i]*dx[i]
        xn[i] = interv(xn[i], i)

    dxn = gradient(xn)
    for i in range(len(x)):
        gamma[i] = np.abs((xn[i] - x[i])*(dxn[i]-dx[i]))/np.linalg.norm((dxn-dx)**2)

    return xn, dx

def func(x):
    f = 0.7854 * x[0] * x[1] ** 2 * \
        (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934) \
        - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2) \
        + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)
    return f

def penal(x):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')

    g1 = (27 / (x1 * x3 * x2 ** 2)) - 1
    g2 = ((397.5) / x1 * x3 * x2 ** 2) - 1
    g3 = ((1.93 * x4 ** 3) / x2 * x3 * x6 ** 2) - 1
    g4 = ((1.93 * x5 ** 3) / x2 * x3 * x7 ** 4) - 1
    g5 = ((745 * x4 / x2 * x3) ** 2 + 16.9 * 10 ** 6)**(1/2) / (110 * x6 ** 3) - 1
    g6 = (((745 * x5) / (x2 * x3)) ** 2 + 157.5 * 10 ** 6)**(1/2) / (85 * x7 ** 3) - 1
    g7 = x2 * x3 / 40 - 1
    g8 = 5 * x2 / x1 - 1
    g9 = x1 / (12 * x2) - 1
    g10 = (1.5 * x6 + 1.9) / x4 - 1
    g11 = (1.1 * x7 + 1.9) / x5 - 1
    g = g1 + g2 + g3 + g4 + g5 + g5 + g6 + g7 + g8 + g9 + g10 + g11
    dg1 = lambdify([x1, x2, x3, x4, x5, x6, x7], g.diff(x1), 'numpy')
    dg2 = lambdify([x1, x2, x3, x4, x5, x6, x7], g.diff(x2), 'numpy')
    dg3 = lambdify([x1, x2, x3, x4, x5, x6, x7], g.diff(x3), 'numpy')
    dg4 = lambdify([x1, x2, x3, x4, x5, x6, x7], g.diff(x4), 'numpy')
    dg5 = lambdify([x1, x2, x3, x4, x5, x6, x7], g.diff(x5), 'numpy')
    dg6 = lambdify([x1, x2, x3, x4, x5, x6, x7], g.diff(x6), 'numpy')
    dg7 = lambdify([x1, x2, x3, x4, x5, x6, x7], g.diff(x7), 'numpy')


    grad = np.array([dg1(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dg2(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dg3(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dg4(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dg5(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dg6(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                     dg7(x[0], x[1], x[2], x[3], x[4], x[5], x[6]),])




    g = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

    h1 = (27 / (x[0] * x[2] * x[1] ** 2)) - 1
    h2 = ((397.5) / x[0] * x[2] * x[1] ** 2) - 1
    h3 = ((1.93 * x[3] ** 3) / x[1] * x[2] * x[5] ** 2) - 1
    h4 = ((1.93 * x[4] ** 3) / x[1] * x[5] * x[6] ** 4) - 1
    h5 = (((745 * x[3] / x[1] * x[2]) ** 2 + 16.9 * 10 ** 6)**(1/2) / 110 * x[5] ** 3) - 1
    h6 = (((745 * x[4]) / (x[1] * x[2])) ** 2 + 157.5 * 10 ** 6)**(1/2) / (85 * x[6] ** 3) - 1
    h7 = x[1] * x[2] / 40 - 1
    h8 = 5 * x[1] / x[0] - 1
    h9 = x[0] / (12 * x[1]) - 1
    h10 = (1.5 * x[5] + 1.9) / x[3] - 1
    h11 = (1.1 * x[6] + 1.9) / x[4] - 1
    h = np.array([h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11])

    if max(h) > 0:
         penalty = grad
    else:
         penalty = 0
    return penalty

def interv(x, i):
    n = np.arange(17,29)
    if i == 2:
        x = int(x)
    switch = {
        0: [2.6, 3.6],
        1: [0.7, 0.8],
        2: [17, 28],
        3: [7.3, 8.3],
        4: [7.8, 8.3],
        5: [2.9, 3.9],
        6: [5, 5.5]
    }
    m = switch.get(i)
    if x>m[1]:
         x = m[1]

    elif x<m[0]:
        x = m[0]

    return x

#Driver
x, dx = gradientDescent(x,iterations, learningRate)
f = func(x)
j = np.abs(dx)
i= 0
while int(max(j)) != 0:
#for i in range(iterations):
    x, dx = gradientDescent(x,iterations, learningRate)
    f = func(x)
    j = np.abs(dx)
    i = i+1
    if i%50 == 0:
        print(dx)
    if i>10**5:
        break

print(x)
print(f)