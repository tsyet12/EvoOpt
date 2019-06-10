'''Gravitational Search Algorithm
###
###Code and Implementation by Sin Yong, Teng
###
###
###Implemented on 06/06/2019
'''

import numpy as np
import matplotlib.pyplot as mp
import time

class GravitationalSearchAlgorithm():
    def __init__(self,f,x,lb,ub,pop=200, max_gen=500,G0=100,a=20,Kbest_min=2,verbose=True):
        self.f=np.vectorize(f)
        self.x=x
        self.lb=lb
        self.ub=ub
        self.pop=pop
        self.max_gen=max_gen
        self.G=G0*np.exp(-a*np.arange(self.max_gen+1)/(self.max_gen+1))
        self.a=a
        self.Kbest_min=Kbest_min
        self.K=np.rint(np.full(self.max_gen+1,self.pop)-((self.pop-self.Kbest_min)/(self.max_gen))*(np.arange(self.max_gen+1))).astype(int)
        np.random.seed(int(time.time()))
        self.pop_mat=np.tile(self.lb,(pop,1))+np.random.rand(pop,len(x)).astype(np.longdouble)*(np.tile(self.ub,(pop,1))-np.tile(self.lb,(pop,1)))
        self.velocity=np.random.uniform(-1,1,self.pop_mat.shape).astype(np.longdouble)*(np.tile(self.ub,(pop,1))-np.tile(self.lb,(pop,1)))
        
        self.verbose=verbose
        
        self.plotgen=[]
        self.average_fit=[]
        self.best_result=[]
        self.best_domain=[]
        self.overall_best=[]
        self.overall_bestdomain=[]
        self.history1=self.pop_mat[:,0]
        self.history2=self.pop_mat[:,1]
        self.velocity_history=np.copy(self.velocity)
    def solve(self):
        self.evaluate()
        for i in range(self.max_gen+1):
            self.update(generation=i)
            self.evaluate()
            if self.verbose:
                self.log_result(generation=i)
    
    def evaluate(self,initial=False):

        #get fitness of all population
        self.pop_mat_fit=self.f(*self.pop_mat.T)

        #concatenate and sort population by fitness
        temp_mat=np.concatenate((np.asarray(self.pop_mat_fit).reshape(self.pop_mat_fit.shape[0],1),self.pop_mat),axis=1)

        #sort new points by fitness
        temp_mat=(temp_mat[temp_mat[:,0].argsort()])

        #return the sorted values to pop matrix
        self.pop_mat_fit, self.pop_mat= np.copy(temp_mat[:,0]), np.copy(temp_mat[:,1:])
               
        #min max normalization
        temp_mat[:,0]=(temp_mat[:,0]-self.pop_mat_fit[-1])/(self.pop_mat_fit[0]-self.pop_mat_fit[-1]+np.finfo(np.float64).eps)

        #assign mass
        self.mass_mat=temp_mat[:,0]/(temp_mat[:,0].sum()+np.finfo(np.float64).eps)

        
    def update(self, generation):
        #compute Euclidian Distance
        R=np.sqrt(np.sum([(self.pop_mat[:,i,None] - self.pop_mat[:,i])**2 for i in range(len(self.x))],axis=0) )
        #compute displacement
        X=np.subtract(self.pop_mat[np.newaxis,:],self.pop_mat[:,np.newaxis])
        
        
        #Removing masses that dont matter
        screened_mass=np.zeros_like(self.mass_mat)       
        screened_mass[:self.K[generation]]=self.mass_mat[:self.K[generation]]
        screened_mass=np.tile(screened_mass,len(self.x)).reshape(len(self.x),self.pop).T
        
        #Defining the random effects on force/acceleration
        a_rand=np.tile(np.random.rand(R.shape[0]),R.shape[1]).reshape(R.shape)
        v_rand=np.tile(np.random.rand(self.velocity.shape[1]),self.velocity.shape[0]).reshape(self.velocity.shape)

        #pre-calculate 1/R*X
        R=a_rand/(R+np.finfo(np.float64).eps)
        R=np.repeat(R,len(self.x),axis=1).reshape(X.shape)
        
        '''
        DEBUG
        
        print('pop_mat=',self.pop_mat)
        print('screened_mass=',screened_mass)
        print('screened_mass=',screened_mass[0,:])
        print('R=',R)
        print('X',X)
        print('X1',screened_mass*X)
        print('X1',R*screened_mass*X)
        print('X1',self.G[generation]*np.sum(R*screened_mass*X,axis=1))
        
        '''

        #Calculating acceleration by crossing out the mass of attraction body
        a=self.G[generation]*np.sum(R*screened_mass*X,axis=1)
        
        #update velocity and displacement
        self.velocity=v_rand*self.velocity+a
        self.pop_mat=np.clip(self.pop_mat+self.velocity,self.lb,self.ub)

    def log_result(self,generation):
        print("Generation #",generation,"Best Fitness=", self.pop_mat_fit[0], "Answer=", self.pop_mat[0])
        self.plotgen.append(generation)
        self.best_result.append(self.pop_mat_fit[0])
        self.best_domain.append(self.pop_mat[0])
        self.overall_best.append(min(self.best_result))
        if self.overall_best[-1]==self.best_result[-1]:
            self.overall_bestdomain.append(self.best_domain[-1])
        else:
            self.overall_bestdomain.append(self.overall_bestdomain[-1])
        self.average_fit.append(np.average(self.pop_mat_fit))
        
        self.history1=np.concatenate((self.history1,self.pop_mat[:,0]),axis=0)
        self.history2=np.concatenate((self.history2,self.pop_mat[:,1]),axis=0)
        self.velocity_history=np.concatenate((self.velocity_history,self.velocity),axis=0)
        
        if generation==self.max_gen:
            print("Final Best Fitness=",self.overall_best[-1],"Answer=",self.best_domain[-1])
            
            
    def plot_result(self,contour_density=50):
        subtitle_font=16
        axis_font=14
        title_weight="bold"
        axis_weight="bold"
        tick_font=14
		
		
        fig=mp.figure()
        
        fig.suptitle("Gravitational Search Algorithm Optimization", fontsize=20, fontweight=title_weight)
        fig.tight_layout()
        mp.subplots_adjust(hspace=0.3,wspace=0.3)
        mp.rc('xtick',labelsize=tick_font)
        mp.rc('ytick',labelsize=tick_font)
		
        mp.subplot(2,2,1)
        mp.plot(self.plotgen,self.overall_best)
		
        mp.title("Convergence Curve", fontsize=subtitle_font,fontweight=title_weight)
        mp.xlabel("Number of Generation",fontsize=axis_font, fontweight=axis_weight)
        mp.ylabel("Fitness of Best Solution",fontsize=axis_font, fontweight=axis_weight)
        mp.autoscale()
		
        mp.subplot(2,2,2)
        mp.plot(self.plotgen,[x[0] for x in self.overall_bestdomain])
        mp.title("Trajectory in the First Dimension",fontsize=subtitle_font,fontweight=title_weight)
        mp.xlabel("Number of Generation", fontsize=axis_font, fontweight=axis_weight)
        mp.ylabel("Variable in the First Dimension",fontsize=axis_font, fontweight=axis_weight)
        mp.autoscale()
        
        
        mp.subplot(2,2,3)
        mp.plot(self.plotgen, self.average_fit)
        mp.title("Average Fitness during Convergence",fontsize=subtitle_font,fontweight=title_weight)
        mp.xlabel("Number of Generation",fontsize=axis_font, fontweight=axis_weight)
        mp.ylabel("Average Fitness of Population",fontsize=axis_font, fontweight=axis_weight)
        
        
        mp.subplot(2,2,4)      
        cont_x=[]
        cont_y=[]
        cont_z=[]
        tempx=self.lb[0]
        tempy=self.lb[1]
        average_other=list(map(lambda x,y:(x+y)/2,self.lb[2:],self.ub[2:]))
        for x in range(contour_density):
            tempx=tempx+(self.ub[0]-self.lb[0])/contour_density
            tempy=tempy+(self.ub[1]-self.lb[1])/contour_density
            cont_x.append(tempx)
            cont_y.append(tempy)
            
        for y in cont_y:
            cont_z.append([self.f(x,y,*average_other) for x in cont_x])            
        mp.plot(self.history1,self.history2,'bo', markersize=3, alpha=0.4)
        CS=mp.contour(cont_x,cont_y,cont_z)
        mp.clabel(CS,inline=True,inline_spacing=-10,fontsize=10)
        mp.title("Points Evaluated",fontsize=subtitle_font,fontweight=title_weight)
        mp.ylabel("Second Dimension",fontsize=axis_font, fontweight=axis_weight)
        mp.xlabel("First Dimension",fontsize=axis_font, fontweight=axis_weight)
        mp.autoscale()
        
        try:
            mng = mp.get_current_fig_manager()
            mng.window.state('zoomed')
        except:
            print("Format your plot using: matplotlib.rcParams['figure.figsize'] = [width, height]")
        mp.show()
        

if __name__=="__main__":

    def f(x1,x2,x3):
        return (x2+(x1-50)**2)/(x3+5)
      
    #(self,f,x,lb,ub,pop=200, max_gen=50,G0=100,a=20,Kbest_min=1,verbose=True)
    
    gsa=GravitationalSearchAlgorithm(f,["x1","x2",'x3'],[0,0,0],[100,100,100],pop=200,max_gen=500,G0=100,a=20,Kbest_min=1,verbose=True)
    gsa.solve()
    gsa.plot_result()   
    
       
        

