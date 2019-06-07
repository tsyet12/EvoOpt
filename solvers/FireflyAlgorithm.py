'''Firefly Algorithm
###
###Code and Implementation by Sin Yong, Teng
###
###
###Implemented on 1/06/2019
'''

import numpy as np
import matplotlib.pyplot as mp
import time

class FireflyAlgorithm():
    def __init__(self,f,x,lb,ub,pop=200, max_gen=50,alpha=0.2,gamma=1,B0=1,Bmin=0.2, d=0.97,gaussian_steps=True,approximate_exp=False, initial_correction=True,verbose=True):
        self.f=np.vectorize(f)
        self.x=x
        self.lb=lb
        self.ub=ub
        self.pop=pop
        self.max_gen=max_gen
        self.Bmin=Bmin
        
        self.initial_correction=initial_correction
        self.alpha=(np.full(self.max_gen+1,alpha)*d**(np.arange(self.max_gen+1)))
        self.gamma=gamma
        self.B0=B0
        self.gaussian_steps=gaussian_steps
        self.d=d
        self.approximate_exp=approximate_exp
        
        np.random.seed(int(time.time()))
        self.pop_mat=np.tile(self.lb,(pop,1))+np.random.rand(pop,len(x)).astype(np.longdouble)*(np.tile(self.ub,(pop,1))-np.tile(self.lb,(pop,1)))
        self.velocity=np.random.uniform(-1,1,self.pop_mat.shape).astype(np.longdouble)*(np.tile(self.ub,(pop,1))-np.tile(self.lb,(pop,1)))
        self.verbose=verbose
        
        self.scale=np.tile(np.asarray(self.ub),self.pop).reshape(self.pop_mat.shape)-np.tile(np.asarray(self.lb),self.pop).reshape(self.pop_mat.shape)
        self.scale_low=np.tile(np.asarray(self.lb),self.pop).reshape(self.pop_mat.shape)
        self.plotgen=[]
        self.average_fit=[]
        self.best_result=[]
        self.best_domain=[]
        self.overall_best=[]
        self.overall_bestdomain=[]
        self.history1=self.pop_mat[:,0]
        self.history2=self.pop_mat[:,1]
        
    def solve(self):
        self.evaluate()
        for i in range(self.max_gen+1):
            self.update(generation=i)
            self.evaluate()
            if self.verbose:
                self.log_result(i)
            
    
    def evaluate(self,initial=False):
        #print('pop',self.pop_mat)
        #get fitness of all population
        self.pop_mat_fit=self.f(*self.pop_mat.T)
        #concatenate and sort population by fitness
        temp_mat=np.concatenate((np.asarray(self.pop_mat_fit).reshape(self.pop_mat_fit.shape[0],1),self.pop_mat),axis=1)
        
        #sort new points by fitness
        temp_mat=temp_mat[temp_mat[:,0].argsort()]  
        
        #return the sorted values to pop matrix
        self.pop_mat_fit, self.pop_mat= np.copy(temp_mat[:,0]), np.copy(temp_mat[:,1:])
        
    def update(self,generation):

        R=np.sum([(self.pop_mat[:,i,None] - self.pop_mat[:,i])**2 for i in range(len(self.x))],axis=0) 
        Intensity=np.tile(self.pop_mat_fit,self.pop).reshape(self.pop,self.pop)
         
        '''
        ###DEBUG###
        print('R',R)
        print('I',Intensity)
        print('IT',Intensity.T)
        print('corrected R=', np.exp(-self.gamma*R))
        print('corrected IR=', Intensity * np.exp(-self.gamma*R))
        '''
        
        Intensity_condition=((Intensity.T )<Intensity).astype(int)
        X=np.subtract(self.pop_mat[np.newaxis,:],self.pop_mat[:,np.newaxis])

        if self.gaussian_steps:
            rand=(np.random.normal(loc=0,scale=0.5,size=self.pop_mat.shape))*self.scale
        else:
            rand=(np.random.rand(*self.pop_mat.shape)-0.5)*self.scale
        
        if self.initial_correction:
            correction=(np.sum((self.B0-self.Bmin)*(R)+self.Bmin,axis=1))
        else:
            correction=0
            
        if self.approximate_exp:
            R=1/(1+self.gamma*np.repeat(R*Intensity_condition,len(self.x),axis=1).reshape(X.shape))
            if self.initial_correction:
                self.pop_mat=(np.sum((self.B0-self.Bmin)*(R)+self.Bmin,axis=1))*self.pop_mat+np.sum(((self.B0-self.Bmin)*(R)+self.Bmin)*X,axis=1)+self.alpha[generation]*(rand)
            else:
                self.pop_mat=self.pop_mat+np.sum(((self.B0-self.Bmin)*(R)+self.Bmin)*X,axis=1)+self.alpha[generation]*(rand)
        else:
            R=np.repeat(np.exp(-self.gamma*R)*Intensity_condition,len(self.x),axis=1).reshape(X.shape)
            if self.initial_correction:
                self.pop_mat=(np.sum((self.B0-self.Bmin)*(R)+self.Bmin,axis=1))*self.pop_mat+np.sum(((self.B0-self.Bmin)*(R)+self.Bmin)*X,axis=1)+self.alpha[generation]*(rand)
            else:
                self.pop_mat=self.pop_mat+np.sum(((self.B0-self.Bmin)*(R)+self.Bmin)*X,axis=1)+self.alpha[generation]*(rand)            
        self.pop_mat=np.clip(self.pop_mat,self.lb,self.ub)

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
        if generation==self.max_gen:
            print("Final Best Fitness=",self.overall_best[-1],"Answer=",self.best_domain[-1])
            
    
    def plot_result(self,contour_density=50):
        subtitle_font=16
        axis_font=14
        title_weight="bold"
        axis_weight="bold"
        tick_font=14
		
		
        fig=mp.figure()
        
        fig.suptitle("Firefly Algorithm Optimization", fontsize=20, fontweight=title_weight)
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
        
        mng = mp.get_current_fig_manager()
        mng.window.state('zoomed')
        mp.show()

if __name__=="__main__":

    def f(x1,x2):
        return ((x1-30)**2+(x2-20)**2+50)

    #    (self,f,x,lb,ub,pop=200, max_gen=50,alpha=0.2,gamma=1,B0=1,verbose=True):
    FA=FireflyAlgorithm(f,["x1","x2"],[-100,-150],[200,200],pop=200,max_gen=50,alpha=0.2,gamma=1,B0=1, Bmin=0, d=0.97,gaussian_steps=True,approximate_exp=False,initial_correction=True,verbose=True)
    FA.solve()
    FA.plot_result()   

       
        

