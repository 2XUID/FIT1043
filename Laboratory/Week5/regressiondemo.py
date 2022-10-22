__author__ = 'Wray Buntine'
_VERSION = '0.1'

import numpy;
import random;

##########
#  globals
#
#  WARNING:  need to recode to remove dependence of beta on x-range!!!
##########
#   x range ... don't change
xmin = 0
xmax = 10
#   smoothing parameter for Smoothed Legendre Polynomials ... pairs with x-range!!
beta = 0.00001
#   scale parameter for noise
sigma = 0.2

##############################
#  support for making datasets
##############################
def setSigma(noise):
    global sigma
    sigma = noise

#  shift X domain to be in [-1,1]
def xshift(x):
    return ((x-xmin)/(xmax-xmin)*2.0 - 1.0)

# have to generate an x sample several times
def makeX(pts,uniform=False):
    if uniform:
        xm = numpy.linspace(xmin, xmax, pts)
    else:
        xm = xmin + numpy.sort(numpy.random.random([pts]))*(xmax-xmin)
    return xm

# here is a noise function
def addNoise(yt,laplace=False):
    if laplace:
        y = yt + sigma/2.0*numpy.random.laplace(size=yt.size)
    else:
        y = yt + sigma*numpy.random.randn(len(yt))
    return y

#  build a set of polynomial orders to use to fit
def makeOrders(points):
    orders = [int(points/1.2)]
    if orders[0]>200:
        orders[0] = 200
    while orders[0]/2>2:
        orders.insert(0,int(orders[0]/2))
    return orders

def maxOrder(points):
    order = int(points/1.2)
    if order>200:
        order = 200
    return order

####################
#  simple regression
####################
#  stops polyfit complaining about poor matrix
import warnings
warnings.simplefilter('ignore', numpy.RankWarning)

#  return yts for poly fit to (xs,ys)
def linReg(xs,ys,xts,order):
     # build the polynomial, a Python object from numpy
     polynomial = numpy.poly1d(numpy.polyfit(xs, ys, order))

     #  build the fitted poly curve (xts,???) from the polynomial
     return polynomial(xts)

#  return a polynomial that is about as good as you get
def bestLinReg(func,xts,order):
    # fit with large amount of data and no noise
    numpts = order*50
    xb = numpy.linspace(xmin, xmax, numpts)
    yb = func(xb)
    return linReg(xb, yb, xts, order) 


################################
#  smoothed Legendre polynomials
################################
from numpy.polynomial import legendre

class LegPoly:
  #  make a Smoothing matrix
  def smoother(self):
    dim = self.order
    A = numpy.zeros((dim+1))
    B = numpy.zeros((dim+1))
    C = numpy.zeros((dim+1))
    for n in numpy.arange(2,dim+1):
        A[n] = (2*n-1)*(2*n-3)
    for n in numpy.arange(4,dim+1):
        B[n] = 2*(2*n-3)*(2*n-7)
    for n in numpy.arange(6,dim+1):
        C[n] = (2*n-5)*(2*n-11)
    self.smooth = numpy.zeros((dim+1,dim+1))
    self.smooth[0][0] = 0.01  # make these smaller since not so fussed
    self.smooth[1][1] = 0.01
    for n in numpy.arange(2,dim+1):
        self.smooth[n][n] += A[n]*A[n] + B[n]*B[n] + C[n]*C[n]
    for n in numpy.arange(2,dim-1):
        self.smooth[n][n+2] += A[n]*B[n+2] + B[n]*C[n+2]
        self.smooth[n+2][n] = self.smooth[n][n+2]
    for n in numpy.arange(2,dim-3):
        self.smooth[n][n+4] += A[n]*C[n+4]
        self.smooth[n+4][n] = self.smooth[n][n+4]
    
  def __init__(self, order,lmbda=beta):
    self.order = order
    #  initialise smoothing matrix
    self.smoother()
    #  tradeoff parameter fer smoother
    self.lmbda = lmbda
    #  hyperparameter, prior for self.lmbda
    self.beta = lmbda
    #  vector giving the polynomial
    self.coeff = numpy.zeros(order+1)
    #  Vandermonde matrix, saved in some cases
    self.vander = None
    self.x = None
    
  def apply(self, x):
    xs = numpy.copy(x)
    xs = xshift(xs)
    return legendre.legval(xs, self.coeff)

    # store x and compute the Vandermonde matrix
  def setX(self, x):
    self.x = numpy.copy(x)
    self.x = xshift(self.x)
    self.vander = legendre.legvander(self.x, self.order)
 
    #   equivalent to polyfit() but using smoothed Legendre polynomials
    #   also calls setX()
  def fit(self, x,y):
    self.setX(x)
    VtV = numpy.dot(self.vander.T,self.vander)
    #  should do SVD for robustness, but this is a simple demo
    legcoeff = numpy.dot( numpy.linalg.inv(VtV + sigma*sigma*self.lmbda*self.smooth), numpy.dot(y, self.vander))
    #  estimate lambda
    # lambda is posterior Gamma(alpha+order/2, beta + cosq/2)
    cosq = numpy.dot(legcoeff.T,numpy.dot(self.smooth,legcoeff))
    self.lmbda = (2+self.order)/(2/self.beta+cosq)
    #  recompute based on estimate of lambda
    legcoeff = numpy.dot( numpy.linalg.inv(VtV + sigma*sigma*self.lmbda*self.smooth), numpy.dot(y, self.vander))   
    # save the value
    self.coeff = numpy.array(legcoeff).flatten()

    #  an MCMC sampler working off previously stored value
    #  samples both self.coeff and self.lmbda
  def sample(self, y):
    # self.x, self.vander, self.coeff already set
    VtV = numpy.dot(self.vander.T,self.vander)/(sigma*sigma) + self.lmbda*self.smooth
    #  do SVD as we want a sqrt of the matrix
    Vu, Vs, Vv = numpy.linalg.svd(VtV)
    #  inefficient because we also compute inverse here!
    legcoeff = numpy.dot( numpy.linalg.inv(VtV)/(sigma*sigma), numpy.dot(y, self.vander))
    self.coeff = numpy.array(legcoeff).flatten() + numpy.dot(Vu,1/numpy.sqrt(Vs)*numpy.random.normal(0,1,self.order+1))
    cosq = numpy.dot(legcoeff.T,numpy.dot(self.smooth,legcoeff))
    # lambda is posterior Gamma(alpha+order/2, beta + cosq/2)
    self.lmbda = numpy.random.gamma(1+self.order/2, 1.0/(self.beta+cosq/2))
    # print "New lambda = "+str(self.lmbda)
    # print "Mean lambda = "+str((2+self.order)/(2/self.beta+cosq))

def demoReg1(nmpts):
    x = makeX(nmpts)
    y = numpy.sin(0.2 + 6/(1+(x/2.5)**1.5))
    y = addNoise(y)
    return x,y

