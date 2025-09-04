from rmellipse.uobjects import RMEMeas
import xarray as xr
import numpy as np
import time 
import matplotlib.pyplot as plt
import matplotlib as mpl

COEFFS_EXPECTED = [1.0, 2.0, 3.0]

def poly2(x, a,b,c):
    return a*x**2  + b*x + c

def setup_fitting(N = 10):
    np.random.seed(0)
    x = np.linspace(0,1,N)
    y = (x**2 + 2*x + 3) + (np.random.random(N)-0.5)*0.01
    y = RMEMeas.from_nom(
        'y',
        xr.DataArray(y, coords = {'x':x}, dims = ('x'))
    )
    y.add_umech('test', value = y.nom + 0.0001)
    y.add_umech('test2', value = y.nom + 0.0015)
    return x, y

def assert_fitting(coeffs):
    assert (np.round(coeffs.nom,2) == np.round(COEFFS_EXPECTED )).all()

def test_curvefit():
    x, y = setup_fitting()

    coeffs = y.curvefit('x', poly2)
    print(coeffs, coeffs.stdunc().cov)
    y_fit = coeffs.curveval(poly2, y.nom.x)
    print(y_fit.nom - y.nom)
    assert_fitting(coeffs)


def speed_test_curvefit(N = 1000, plot = False):
    x, y = setup_fitting(N = N)
    t0 = time.time()
    for i in range(N):
        y.add_umech(f'mech{i}',y.nom+(np.random.random(y.nom.shape)-.5)*0.5)
    print('setup_time:', (time.time() - t0)/60)
    t1 = time.time()
    coeffs = y.curvefit('x', poly2)
    print('fit_time:', (time.time() - t1)/60)
    values = coeffs.curveval(poly2, y.nom.x)
    if plot:
        # mpl.use('qtagg')
        plt.close('all')
        fig,ax = plt.subplots(1,1)
        ax.errorbar(x, y.nom, yerr = y.stdunc().cov, fmt = 'ko')
        ax.plot(values.nom.x,values.nom, 'b')
        ax.plot(values.nom.x,values.uncbounds(k = 1).cov,'r--')
        ax.plot(values.nom.x,values.uncbounds(k = -1).cov,'r--')
        plt.show()
    print(values, values.stdunc().cov)


def test_polyfit():
    x, y = setup_fitting()

    coeffs = y.polyfit('x',2)
    print(coeffs)
    y_fit = coeffs.polyval(y.cov.x, 'degree')
    print(y_fit)
    assert_fitting(coeffs)


if __name__ == '__main__':
    test_polyfit()
    test_curvefit()
    speed_test_curvefit(N = 10, plot = True)