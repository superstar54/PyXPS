# -*- coding: utf-8 -*-

""" 
Generate spectrum databases for deep learning training and testing.
Authers: Xing Wang (xingwang1991@gmail.com)
"""
import numpy as np
from numpy import sqrt, pi, exp, linspace, random
import matplotlib.pyplot as plt



class spectrum(object):
    """Spectrum
       Initial the spectrum with zeros arrays;
       Add curves and background to the spectrum;
       Two kinds of curves: gaussian and lorentz;

       #Backgroud
       (-1)**(i+1)*k[i]*(x - xmean)**i
       e.g. -k1 + k2*(x - xmean) - k3*(x - xmean)**2
    """
    def __init__(self, numx=101, x_range=[1200, 1000]):
        self.numx = numx
        self.npeaks = 0
        self.x = np.linspace(x_range[0], x_range[1], self.numx)
        self.y = {}
        # Gaussian
        self.y['gaussian'] = np.zeros(self.numx)
        self.gpara = {'cen': [], 'amp': [], 'width': []}
        # Gaussian
        self.y['lorentzian'] = np.zeros(self.numx)
        self.lpara = {'cen': [], 'gamma': [], 'width': []}
        # background
        self.backg = np.zeros(self.numx)
    #
    def generate(self):
        y = self.gaussian(self.x, self.amp, self.cen, self.width) + random.normal(0, self.amp/50, self.numx)
        self.y['gaussian'] = y
    #    
    def add_lorentzian(self, x, gamma, width, cen):
        y = gamma*width/((x-cen)**2 + width**2)
        self.y['lorentzian'] = y
    #
    def add_backg(self, ks):
        mean = np.mean(self.x)
        for i in range(len(ks)):
            self.backg += (-1)**(i+1)*ks[i]*(self.x - mean)**i
        self.y['gaussian'] += self.backg
    #
    def add_gaussian(self, amp, cen, wid, noise = 0.1):
        self.gpara['cen'].append(cen)
        self.gpara['amp'].append(amp)
        self.gpara['width'].append(wid)
        y = amp * exp(-(self.x-cen)**2 /wid) + random.normal(0, amp*0.02, self.numx)
        self.y['gaussian'] += y
        self.npeaks += 1
    #
    def show(self):
        plt.plot(self.x, self.y['gaussian'], 'b-')
        plt.show()
        # plt.imshow(self.y[])
    #
    def savefig(self, finame):
        plt.figure()
        plt.plot(self.x, self.y['gaussian'], 'b-')
        plt.savefig(finame)
        # plt.imshow(self.y[])


if __name__ == "__main__":
    nums = 10
    numx = 50
    npks = 3
    spectrums = []
    amp=random.uniform(2000, 6000, nums)
    cen=random.uniform(1020, 1180, nums)
    wid=random.uniform(20, 50, nums)
    npeaks = random.randint(1, npks + 1, nums)
    for i in range(nums-npks):
        xps = spectrum(numx)
        for j in range(npeaks[i]):
            xps.add_gaussian(amp[i + j], cen[i + j], wid[i + j], noise = 0.005)
        xps.add_backg([1, 5, 0.01])
        spectrums.append(xps)
        print(xps.npeaks)
        # xps.show()
        xps.savefig('figs/{0}_{1}.jpg'.format(i, xps.npeaks))
