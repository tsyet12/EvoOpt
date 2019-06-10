'''Particle Swarm Optimization
###
###Code and Implementation by Sin Yong, Teng
###
###
###Implemented on 1/06/2019
'''


import numpy as np
import matplotlib.pyplot as mp

class ParticleSwarmOptimization():
    def __init__(self,f,x,lb,ub,pop=200, max_gen=50,w=0.9,c1=2,c2=1,verbose=True):
        self.f=np.vectorize(f)
        self.x=x
        self.lb=lb
        self.ub=ub
        self.pop=pop
        self.max_gen=max_gen
        self.w=w
        self.c1=c1
        self.c2=c2
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
        self.evaluate(initial=True)
        self.update_velocity_position()
        for i in range(self.max_gen+1):
            self.update_velocity_position()
            self.evaluate()
            if self.verbose:
                self.log_result(i)
            
    
    def evaluate(self,initial=False):
        #get fitness of all population
        self.pop_mat_fit=self.f(*self.pop_mat.T)

        #concatenate and sort population by fitness
        temp_mat=np.concatenate((np.asarray(self.pop_mat_fit).reshape(self.pop_mat_fit.shape[0],1),self.pop_mat),axis=1)
        
        #update local best position for all searchers
        if initial:
            #initialize local best if it is the 0th iteration
            self.local_best_position=np.copy(temp_mat)
        else:
            #for other iterations, compare if new point is better than local best. If yes, then update local best by new point
            for i in range(self.local_best_position.shape[0]):
                if temp_mat[i,0]<self.local_best_position[i,0]:
                    self.local_best_position[i,:]=temp_mat[i,:]
        
        #sort new points by fitness
        temp_mat=temp_mat[temp_mat[:,0].argsort()]  
        
        sorted_local_best_position=self.local_best_position[self.local_best_position[:,0].argsort()]
        
        #update global best
        self.global_best_fit, self.global_best_domain=sorted_local_best_position[0,0], sorted_local_best_position[0,1:]
        #print(self.global_best_fit)
        
        
    def update_velocity_position(self):
        rp=np.random.rand(*self.pop_mat.shape)
        rg=np.random.rand(*self.pop_mat.shape)
        
        self.velocity=self.w * self.velocity+ self.c1* rp * (self.local_best_position[:,1:]-self.pop_mat) \
        + self.c2* rg * (self.global_best_domain - self.pop_mat)
        #print(self.velocity)
        
        self.pop_mat=self.pop_mat+self.velocity
        self.pop_mat=np.clip(self.pop_mat,np.tile(self.lb,(self.pop_mat.shape[0],1)),np.tile(self.ub,(self.pop_mat.shape[0],1)))
        
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
        
        
            
            

    def plot_result(self,contour_density=50):
        subtitle_font=16
        axis_font=14
        title_weight="bold"
        axis_weight="bold"
        tick_font=14
		
		
        fig=mp.figure()
        
        fig.suptitle("Particle Swarm Optimization", fontsize=20, fontweight=title_weight)
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
        return (x1+x2)/(x3+0.000000000001)
    #(self,f,x,lb,ub,pop=200, max_gen=50,w=0.9,c1=2,c2=1,verbose=True):
    pso=ParticleSwarmOptimization(f,["x1","x2","x3"],[0,0,0],[100,100,100],pop=200,max_gen=200,w=0.9,c1=2,c2=1,verbose=True)
    pso.solve()
    pso.plot_result()   
    
       
        

