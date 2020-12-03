import numpy as np
import time
learningRate = 0.01

iterations = 1000
N = 10
x1 = np.linspace(2.6, 3.6, N) #largura da cremalheira
x2 = np.linspace(0.7, 0.8, N) #módulo da engrenagem
x3 = np.arange(17,29) #nº de dentes do pinhão
x4 = np.linspace(7.3, 8.3, N) #comprimento do primeiro veio entre rolamentos
x5 = np.linspace(7.8, 8.3, N) #comprimento do segundo veio entre rolamentos
x6 = np.linspace(2.9, 3.9, N) #diâmetro do primeiro veio
x7 = np.linspace(5, 5.5, N) #diâmetro do segundo veio

x = np.array([[3.6],
              [0.7],
              [29],
              [7.3],
              [7.8],
              [2.9],
              [5]])
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
    dx1 = x[0]**2*(2.61797*x[2]**2+11.7287*x[2]-33.8456)-1.508*x[5]**2-1.508*x[6]**2
    dx2 = 5.23595*x[0]*x[1]*(x[2]**2+4.48006*x[2]-12.9281)
    dx3 = 5.23595*x[0]*x[1]**2*(x[2]+2.24003)
    dx4 = 0.7854*x[5]**2
    dx5 = 0.7854*x[6]**2
    dx6 = x[5]*(-3.016*x[0] + 1.5708*x[3]+22.4331*x[5])
    dx7 = x[6]*(-3.016*x[0]+1.5708*x[4]+22.4331*x[6])

    grad = np.array([dx1, dx2, dx3, dx4, dx5, dx6, dx7])
    return grad

def gradientDescent(x, iterations, learningRate):
    xn = np.zeros(len(x))
    dxn = np.zeros([len(x), 1])
    gamma = np.ones([len(x), 1]) * 0.001

    dx = gradient(x)

    for i in range(len(x)):
        xn[i] = x[i] - gamma[i]*dx[i]

    dxn = gradient(xn)
    for i in range(len(x)):
        gamma[i] = np.abs((xn[i] - x[i])*(dxn[i]-dx[i]))/np.linalg.norm((dxn-dx)**2)

    return xn

def func(x):
    f = 0.7854 * x[0] * x[1] ** 2 * \
        (3.3333 * x[2] ** 2 + 14.9934 * x[2] - 43.0934) \
        - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2) \
        + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)

def penal():


f = gradientDescent(x,iterations, learningRate)
print(f)