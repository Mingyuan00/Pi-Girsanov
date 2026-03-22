from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
from matplotlib import pyplot as plt

class DoubleWellForce(openmm.CustomExternalForce):
    """1D Double well potential"""
    A = 50
    c1 = -1
    c2 = -1
    mu1 = -0.5
    mu2 = 0.5
    sigma1 = 0.25
    sigma2 = 0.25
    x_range = [-1.0,1.0]

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * y^2 + 1000.0 * z^2 +'
        self.expression = '{A}*({c1}*exp(-(x-{mu1})^2/{sigma1})+{c2}*exp(-(x-{mu2})^2/{sigma2}))'.format(
            A=self.A,c1=self.c1,c2=self.c2,mu1=self.mu1,mu2=self.mu2,sigma1=self.sigma1,sigma2=self.sigma2
        )
        expression += self.expression
        expression += '''+ step({minx}-x) * 1000.0 * x^2 + step(x-{maxx}) * 1000 * x^2'''.format(minx=self.x_range[0],maxx=self.x_range[1])
        super(DoubleWellForce, self).__init__(expression)
        
    @classmethod
    def potential(cls,x):
        "Compute the potential at a given point x"
        value = cls.A*(cls.c1*np.exp(-(x-cls.mu1)**2/cls.sigma1)+cls.c2*np.exp(-(x-cls.mu2)**2/cls.sigma2))
        return value

    @classmethod
    def biased_potential(cls,x,biasfactor):
        "Compute the biased potential at a given point x"
        value = cls.potential(x) + (1/biasfactor - 1) * cls.potential(x)
        return value

    @classmethod
    def plot(cls,biasfactor=None,ax=None,nbins=250,**kwargs):
        "Plot the doublewell potential"
        x = np.linspace(cls.x_range[0],cls.x_range[1],nbins)
        V = cls.potential(x)
        V = V - V.min()
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(x,V,label='unbiased',**kwargs)
        if biasfactor is not None:
            V_biased = cls.biased_potential(x,biasfactor)
            V_biased = V_biased - V_biased.min()
            ax.plot(x,V_biased,label=r'biased $\gamma=${biasfactor}'.format(biasfactor=biasfactor))

class QuadraWellForce(openmm.CustomExternalForce):
    """1D quadra well potential (Prinz potential)"""
    A = 20
    x_range = [-1.0,1.0]

    def __init__(self):
        # start with a harmonic restraint on the Y and Z coordinates
        expression = '1000.0 * y^2 + 1000.0 * z^2 +'
        self.expression = '{A}*(x^8 + 0.8*exp(-80*(x^2)) + 0.2*exp(-80*(x-0.5)^2) + 0.5*exp(-40*(x+0.5)^2))'.format(
            A=self.A
        )
        expression += self.expression
        expression += '''+ step({minx}-x) * 1000.0 * x^2 + step(x-{maxx}) * 1000 * x^2'''.format(
            minx=self.x_range[0],maxx=self.x_range[1]
        )
        super(QuadraWellForce, self).__init__(expression)
        
    @classmethod
    def potential(cls,x):
        "Compute the potential at a given point x"
        value = cls.A*(x**8 + 0.8*np.exp(-80*(x**2)) + 0.2*np.exp(-80*(x-0.5)**2) + 0.5*np.exp(-40*(x+0.5)**2))
        return value

    @classmethod
    def biased_potential(cls,x,biasfactor):
        "Compute the biased potential at a given point x"
        value = cls.potential(x) + (1/biasfactor - 1) * cls.potential(x)
        return value

    @classmethod
    def plot(cls,biasfactor=None,ax=None,nbins=250,**kwargs):
        "Plot the quadrawell potential"
        x = np.linspace(cls.x_range[0],cls.x_range[1],nbins)
        V = cls.potential(x)
        V = V - V.min()
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(x,V,label='unbiased',**kwargs)
        if biasfactor is not None:
            V_biased = cls.biased_potential(x,biasfactor)
            V_biased = V_biased - V_biased.min()
            ax.plot(x,V_biased,label=r'biased $\gamma=${biasfactor}'.format(biasfactor=biasfactor))

class MullerForce(openmm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    #AA = [-30, -15, -25.5, 2.25]
    AA = [-40, -20, -34, 3]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    x_range = [-1.3,1.0]
    y_range = [-0.25,2]

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'
        self.expression = ''
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j], cc=self.cc[j], AA=self.AA[j], XX=self.XX[j], YY=self.YY[j])
            self.expression += '+ {AA}*exp({aa} *(x - {XX})^2 + {bb} * (x - {XX}) * (y - {YY}) + {cc} * (y - {YY})^2)'.format(**fmt)
        # Restraint to confine particles in x_range and y_range
        expression += self.expression
        expression += '''+ step({minx}-x) * 1000.0 * x^2 + step(x-{maxx}) * 1000 * x^2'''.format(minx=self.x_range[0],maxx=self.x_range[1])
        expression += '''+ step({miny}-y) * 1000.0 * y^2 + step(y-{maxy}) * 1000 * y^2'''.format(miny=self.y_range[0],maxy=self.y_range[1])
        super(MullerForce, self).__init__(expression)
        self.expression = self.expression[1:]

    @classmethod
    def potential(cls,x,y):
        "Compute the potential at a given point x"
        value = 0
        for j in range(4):
            value += cls.AA[j] * np.exp(cls.aa[j] * (x - cls.XX[j])**2 + 
                     cls.bb[j] * (x - cls.XX[j]) * (y - cls.YY[j]) + cls.cc[j] * (y - cls.YY[j])**2)
        return value

    @classmethod
    def biased_potential(cls,x,biasfactor):
        "Compute the biased potential at a given point x"
        value = cls.potential(x) + (1/biasfactor - 1) * cls.potential(x)
        return value

    @classmethod
    def plot(cls,ax=None,levels=20,vmax=30,nbins=100,**kwargs):
        "Plot the quadrawell potential"
        x = np.linspace(cls.x_range[0],cls.x_range[1],nbins)
        y = np.linspace(cls.y_range[0],cls.y_range[1],nbins)
        X, Y = np.meshgrid(x, y)
        V = cls.potential(X, Y)
        V = V - V.min()
        if ax is None:
            fig,ax = plt.subplots(figsize=(8,6))
        contourf = ax.contourf(X,Y,V.clip(max=vmax),cmap='jet',levels=levels,**kwargs)
        plt.colorbar(contourf,ax=ax,label='Energy (kJ/mol)',ticks=np.linspace(0,vmax,int(levels/2+1)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
