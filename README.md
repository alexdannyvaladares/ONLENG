# Weight minimisation of a speed reducer

The problem was proposed within the scope of the Non-Linear Otimization curricular unit of the mechanical engineering master. For a detailed explanation of the formulation,code and results discussion, check the Benchmark2020.pdf.

### **Introduction of the Non-Linear Problem**

A speed reducer is simply a gear train between the motor and the machinery that is used to reduce the speed with which power is transmitted. Speed reducers, also called gear reducers, are mechanical gadgets by and large utilised for two purposes. First, they take the torque created by the power source (the input) and multiply it. Second, speed reducers, much as the name implies, reduce the speed of the input so that the output is the correct speed. In other words, gear reducers
essential use is to duplicate the measure of torque produced by an information power source to expand the measure of usable work.

The weight of a speed reducer is to be minimised subjected to constraints on levels of bending stress of the gear teeth, surface stress, transverse deflections of the shafts and stresses in the shafts [2]. This design 2 A. Andrade-Campos, J. Dias-de-Oliveira problem involves seven design variables, as shown in Figure 2, which are the face width, x1, module of the teeth, x2, number of teeth on the pinion, x3, length of the first shaft between bearings, x4, length of the second shaft between bearings, x5, diameter of the first shaft, x6, and diameter of the second shaft, x7. The third variable, x3, is an integer, while the rest are continuous. With eleven constraints, this is a constrained optimisation problem.

![reducer](https://github.com/alexdannyvaladares/ONLENG/blob/main/Benchmark/Reducer.JPG)

### **Optimization algorithm used & challenges**

**Adaptive moment estimation method:**
This is a method that you can see as a combination between the stochastic gradient method and RMSProp. It uses the square of the gradients to update the “learning rate” at each iteration, at the same time that we update moments in order to converge more quickly.

**Challenge :**
The implementation of the algorithm, in python, was a complex task because with the gradient method, there is a need to make the function objective from a restricted function, to an unrestricted function. To this end, we apllied some measures such as:


-Use of a barrier function, in restrictions


-Use of a smoothed module function in domain constraints.
