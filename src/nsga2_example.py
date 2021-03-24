'''
Created on Oct 8, 2020

@author: Nuno Lourenço

Example of using NSGA2 on a dummy function

'''

import numpy as np
import optimizers as opt


class Func1(opt.Problem):
    """
    The problem related parameters and genetic operations
    """
    def __init__(self):
        super().__init__(
            ["f1","f2", "f3"],
            ["x"+str(i) for i in range(10)],
            -np.ones((1, 10)),
            np.ones((1, 10)))

    def cost_fun(self, x):
        """
        calculate the objective vectors
        :param x: the decision vectors
        :return: the objective vectors
        """
        n = x.shape[0]
        a = np.zeros((self.M, self.d))
        
        for i in range(self.d):
            for j in range(self.M):
                a[j,i] = ((i+0.5)**(j-0.5))/(i+j+1.)
        obj = np.zeros((n, self.M))
        meas = np.zeros((n, 1))
        cstr = np.zeros(n)
        for i in range(n):
            for j in range(self.M):
                obj[i, j] = np.dot(x[i, :] ** (j + 1), a[j, :].T)
        return obj, cstr, meas



if __name__ == '__main__':
    seed = 17
    np.random.seed(seed)


    a = opt.Nsga2(Func1(),pop_size=512, evaluations=512*100 )

    pop_dec, pop_obj, pop_cstr, pop_meas = a.run()

    
