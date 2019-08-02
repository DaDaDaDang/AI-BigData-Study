import numpy as np
from scipy.stats import mode
import statistics as sta
import matplotlib.pyplot as plt

def ndarray_structer () :
    npa = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(npa.size)
    print(npa.shape)
    print(len(npa))

    npl = np.array([1, 100, 42, 42, 42, 6, 7])
    print(npl.size)
    print(len(npl))
    print(npl.shape)
    print(npl.ndim)

    print()

def ndarray_create_zeros () :
    a = np.zeros(3)
    print(a.ndim)
    print(a.shape)
    print(a)

    print()

def ndarray_create_eye () :
    np.eye(2, dtype = int)
    print(np.eye(3))
    print(np.eye(3, k=1))
    print(np.eye(3, k=-1))

    print()

def ndarray_create_indentity () :
    print(np.identity(5))

    print()

def ndarray_create_linespace () :
    epTrue = np.linspace(2, 3, num = 5, endpoint=True, retstep=False)
    epFalse = np.linspace(2, 3, num = 5, endpoint=False, retstep=False)
    rtsTrue = np.linspace(2,3,num=5,endpoint=True,retstep=True)
    # a=np.linspace(2,3,num=5,endpoint=True,retstep=False,dtype=class)
    print(epTrue)
    print(epFalse)
    print(rtsTrue)
    # print(a)
    print()

def  ndarray_cal_1 () :
    x = np.array([1, 5, 2])
    y = np.array([7, 4, 1])
    print(x + y)
    print(x * y)
    print(x - y)
    print(x / y)
    print(x % y)

    print()

def  ndarray_cal_2 () :
    bb = np.array([1,2,3])
    cc = np.array([-7, 8, 9])
    print(np.dot(bb,cc))

    xs = np.array(((2,3), (3,5)))
    ys = np.array(((1,2), (5, -1)))
    print(np.dot(xs, ys), type(np.dot(xs, ys)))

    print()

def ndarray_array_access_1() :
    I33 = [[1,2,3],[4,5,6],[7,8,9]]
    np33 = np.array(I33, dtype=int)

    print(np33.shape)
    print(np33.ndim)
    print(np33)
    print("first row : ", np33[0])
    print("first column : ", np33[:, 0])

    print()

def ndarray_array_access_2() :
    I33 = [[1,2,3],[4,5,6],[7,8,9]]
    np33 = np.array(I33, int)
    print(np33)

    print(np33[:2, 1:])

    print()

def ndarray_array_access_3() :
    arr = np.array([9, 18, 29, 39, 49])

    print(" index ")
    print(arr.argmax())
    print(arr.argmin())

    print(" value ")
    print(arr[np.argmax(arr)])
    print(arr[np.argmin(arr)])

    print()

def ndarray_Axis () :
    a = np.arange(6)
    b = np.arange(6).reshape(2, 3)
    a[5] = 100

    print(a)
    print(b)
    print(a[np.argmax(a)])

    print(np.argmax(b, axis=0))
    print(np.argmax(b, axis=1))

    print()

def ndarray_rand_2 () :
    a = np.random.rand(3, 2)
    print(a)

    b = np.random.rand(3,3,3)
    print(b)

    print()

def ndarray_rand_3() :
    outcome = np.random.randint(1, 7, size=10)
    print(outcome)
    print(type(outcome))
    print(len(outcome))

    print(np.random.randint(2, size=10))
    print(np.random.randint(1, size=10))
    print(np.random.randint(5, size=(2, 4)))

    print()

def ndarray_rand_4():
    a = np.random.randn(3, 2)  # 이거 달아주셈 import matplotlib.pyplot as plt
    print(a)
    b = np.random.randn(3, 3, 3)
    print(b)
    plt.plot(a)
    plt.show()

    print()

def ndarray_rand_5():
    arr = np.arange(10)
    print(arr)
    np.random.shuffle(arr)
    print(arr)

    arr2 = np.arange(9).reshape((-1, 3))
    print(arr2)
    np.random.shuffle(arr2)
    print(arr2)

    print()

def ndarray_basic_stat_1():
    x = np.array([-2.1, -1, 1, 1, 4.3])
    print(np.mean(x))
    print(np.median(x))
    print(mode(x))

    print()

def ndarray_basic_stat_2():
    x = np.array([-2.1, -1, 1, 1, 4.3])
    print(np.mean(x))
    print(np.median(x))
    print(mode(x))

    x_m = np.mean(x)
    x_a = x - x_m
    x_p = np.power(x_a, 2)

    print("Variance x")
    print("np.var(x)")
    print(sta.pvariance(x))
    print(sta.variance(x))

    print()

def ndarray_basic_stat_3 () :
    x = np.array([-2.1, -1, 1, 1, 4.3])
    print(np.mean(x))
    print(np.median(x))
    print(mode(x))

    x_m = np.mean(x)
    x_a = x - x_m
    x_p = np.power(x_a, 2)
    print(np.var(x))
    print(sta.pvariance(x))
    print(sta.variance(x))

    print(np.std(x))
    print(sta.pstdev(x))
    print(sta.stdev(x))

    print()

if __name__ == '__main__' :
    ndarray_structer()
    ndarray_create_zeros()
    ndarray_create_eye()
    ndarray_create_indentity()
    ndarray_create_linespace()
    ndarray_cal_1()
    ndarray_cal_2()
    ndarray_array_access_1()
    ndarray_array_access_2()
    ndarray_array_access_3()
    ndarray_Axis()
    ndarray_rand_2()
    ndarray_rand_3()
    ndarray_rand_4()
    ndarray_rand_5()
    ndarray_basic_stat_1()
    ndarray_basic_stat_2()
    ndarray_basic_stat_3()