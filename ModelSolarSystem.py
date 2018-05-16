#Author: Alexander Criswell

#Code to set up and run an n-body 2D solar system
#import numpy, scipy
import numpy as np
import scipy.integrate as spi
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
#Constants
au = 1.496*10**11 #m
msun = 1.989*10**30 #kg
me = 5.972*10**24 #kg
yr = 3.154*10**7 #s
G = 6.67*10**-11 #Gravitational constant
#Create body class for sun and planets
#name = name
#mass = mass
#ti = initial period
#ai = initial semi-major axis
#ei = initial eccentricity
#thi = initial angle

class Body(object):
    def __init__(self,name,m,ti,ai,ei,thi):
        self.name = name
        self.mass = m
        self.per = ti
        self.sma = ai
        self.ecc = ei
        self.theta = thi
        self.r = ai*(1-ei**2)/(1+ei*np.cos(thi))
        self.anomaly = np.arccos((1/self.ecc)*(1-(self.r/self.sma)))
        self.vx = ((np.sqrt(G*(1.989*10**30)*self.sma)/self.r)*(-1*np.sin(self.anomaly)))
        self.vy = ((np.sqrt(G*(1.989*10**30)*self.sma)/self.r)*(np.sqrt(1-(self.ecc)**2)*np.sin(self.anomaly)))
    def getpc(self):
        return np.array([self.r,self.theta])
    def getcc(self):
        return np.array([self.r*np.cos(self.theta),self.r*np.sin(self.theta)])
    def updatexy(self,xc,yc):
        self.r = np.sqrt(xc**2 + yc**2)
        self.theta = np.tan(yc/xc)
    def updatev(self,vx,vy):
        self.vx = vx
        self.vy = vy
        
    

#Create Sun

class Sun(Body):
    def __init__(self,name,m,ti,ai,ei,thi):
        ai = 1 #ai and ei are arbitrary constants to avoid divide by zero errors
        ei = 0.5
        super().__init__(name,m,ti,ai,ei,thi)
        self.anomaly = 0
        self.vx = 0
        self.vy = 0
        self.ecc = 0
        self.sma = 0
        self.r = 0

sun = Sun('Sun',msun,0,0,0,0)


#Create rest of bodies (Data from NASA Planetary Fact Sheets)
mercury = Body('Mercury',.0553*me,.241*yr,.387*au,.205,np.random.rand()*2*np.pi)
venus = Body('Venus',.815*me,.615*yr,.723*au,.007,np.random.rand()*2*np.pi)
earth = Body('Earth',me,yr,au,.017,np.random.rand()*2*np.pi)
mars = Body('Mars',.107*me,1.88*yr,1.52*au,.094,np.random.rand()*2*np.pi)
jupiter = Body('Jupiter',317.8*me,11.9*yr,5.20*au,.049,np.random.rand()*2*np.pi)
saturn = Body('Saturn',95.2*me,29.4*yr,9.58*au,.057,np.random.rand()*2*np.pi)
uranus = Body('Uranus',14.5*me,83.7*yr,19.20*au,.046,np.random.rand()*2*np.pi)
neptune = Body('Neptune',17.1*me,163.7*yr,30.05*au,.011,np.random.rand()*2*np.pi)


bodies = [sun,mercury,venus,earth,mars,jupiter,saturn,uranus,neptune]
#Find interplanetary distances
distances = {}
ccarr = np.array([i.getcc() for i in bodies])
for b in bodies:
    bd = dist.cdist(np.array([b.getcc()]),ccarr)
    distances[b.name] = np.reshape(bd,len(bodies))


orbits = {}
for b in bodies:
    orbits[b.name] = b.getcc()
print(orbits)

#Numerical Integrator Functions
def accintfun(body,otherbody):
    dx = body.getcc()[0] - otherbody.getcc()[0]
    dy = body.getcc()[0] - otherbody.getcc()[0]
    ax = G*otherbody.mass*((distances[body.name][bodies.index(otherbody)])**(-3))*dx
    ay = G*otherbody.mass*((distances[body.name][bodies.index(otherbody)])**(-3))*dy
    return np.array([ax,ay])

def netaccfun(body):
    anet = np.array([0.,0.])
    for otherbody in bodies:
        if otherbody != body:
            anet += accintfun(body,otherbody)
        else: 
            anet += 0
    return anet 

def velintfun(body):
    vx = np.array([body.vx])
    vy = np.array([body.vy])
    a = netaccfun(body)
    def vxfun(t,vx): return a[0]
    def vyfun(t,vy): return a[1]
    vxsol = spi.solve_ivp(vxfun,[t1,t2],vx)
    vysol = spi.solve_ivp(vyfun,[t1,t2],vy)
    vxf = vxsol.y[0][-1]
    vyf = vysol.y[0][-1]
    body.updatev(vxf,vyf)
    return np.array([vxf,vyf])

def posintfun(body):
    x = np.array([body.getcc()[0]])
    y = np.array([body.getcc()[1]])
    v = velintfun(body)
    def xfun(t,x): return v[0]
    def yfun(t,x): return v[1]
    xsol = spi.solve_ivp(xfun,[t1,t2],x)
    ysol = spi.solve_ivp(yfun,[t1,t2],y)
    xf = xsol.y[0][-1]
    yf = ysol.y[0][-1]
    body.updatexy(xf,yf)
    return np.array([xf,yf])

t1 = 0
tstart = 10**-10
tend = 10**-8
tstep = 10**-10
times = np.arange(tstart,tend,tstep)
for t in times:
    t2 = t
    for b in bodies:
        xy = posintfun(b)
        orbits[b.name] = np.vstack((orbits[b.name],xy))
    t1 = t2
#plt.plot(orbits['Earth'][:,[0]],orbits['Earth'][:,[1]])
#plt.scatter(orbits['Sun'][:,[0]],orbits['Sun'][:,[1]])

for b in bodies:
    plt.scatter(orbits[b.name][:,[0]],orbits[b.name][:,[1]])

plt.axis([-4*10**12,4*10**12,-4*10**12,4*10**12])
plt.show()
#
#    


















