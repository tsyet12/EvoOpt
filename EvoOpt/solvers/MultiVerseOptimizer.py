'''Multi-Verse Optimizer
###
###Code and Implementation by Sin Yong, Teng
###
###
###Implemented on 09/06/2019
'''


import numpy as np
import matplotlib.pyplot as mp

class MultiVerseOptimizer():
    def __init__(self,f,x,lb,ub,pop=200,max_gen=50,min=0.2,max=1,p=6,verbose=True):
        self.f=np.vectorize(f)
        self.x=x
        self.lb=lb
        self.ub=ub
        self.pop=pop
        self.verbose=verbose

        self.max_gen=max_gen
        self.pop_mat=np.tile(self.lb,(pop,1))+np.random.rand(pop,len(x)).astype(np.longdouble)*(np.tile(self.ub,(pop,1))-np.tile(self.lb,(pop,1)))
                
        self.WEP=np.full(self.max_gen+1,min)+np.arange(self.max_gen+1)*(max-min)/max_gen
        self.TDR=1-(np.arange(self.max_gen+1)/max_gen)**(1/p)
        self.vec_ub=np.tile(self.ub,(pop,1))
        self.vec_lb=np.tile(self.lb,(pop,1))
        self.plotgen=[]
        self.average_fit=[]
        self.history1=self.pop_mat[:,0]
        self.history2=self.pop_mat[:,1]
        self.best_result=[]
        self.best_domain=[]
        self.overall_best=[]
        self.overall_bestdomain=[]
        
    def solve(self):
        self.evaluate(initial=True)
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
        temp_mat=temp_mat[temp_mat[:,0].argsort()]  
        
        #return the sorted values to pop matrix
        self.pop_mat_fit, self.pop_mat= np.copy(temp_mat[:,0]), np.copy(temp_mat[:,1:])       
        
        #return normalized value to NI
        self.NI=(self.pop_mat_fit-min(self.pop_mat_fit))/(max(self.pop_mat_fit)-min(self.pop_mat_fit)+np.finfo(np.float64).eps)
    
    @staticmethod
    def RouletteWheelSelection(population_mat):
        random=np.random.randint(low=0,high=population_mat.shape[0],size=population_mat.shape)
        output=np.asarray([[population_mat[random[i,j],j] for j in range(random.shape[1])] for i in range(random.shape[0])]).reshape(population_mat.shape)
        return output
    
    def update(self,generation):
        best_universe=np.tile(self.pop_mat[0,:],self.pop).reshape(self.pop_mat.shape)
        r1=np.random.rand(*self.pop_mat.shape) #whitehole
        newNI=np.repeat(self.NI,len(self.x)).reshape(self.pop_mat.shape)
        whitehole_condition=(r1<newNI).reshape(self.pop_mat.shape)        
        roulette_pop=self.RouletteWheelSelection(self.pop_mat)
        self.pop_mat=whitehole_condition.astype(int)*roulette_pop+(~whitehole_condition).astype(int)*self.pop_mat  
        r2=np.random.rand(*self.pop_mat_fit.shape)  #worm hole
        r3=np.random.rand(*self.pop_mat_fit.shape)
        r4=np.random.rand(*self.pop_mat.shape)
        wormhole_condition=np.repeat(r2<self.WEP[generation],len(self.x)).reshape(self.pop_mat.shape)
        travel_condition=np.repeat(r3<0.5,len(self.x)).reshape(self.pop_mat.shape)

        self.pop_mat=(wormhole_condition).astype(int)*(best_universe+(travel_condition).astype(int)*self.TDR[generation]*((self.vec_ub-self.vec_lb)*r4+self.vec_lb) \
        -(~travel_condition).astype(int)*self.TDR[generation]**((self.vec_ub-self.vec_lb)*r4+self.vec_lb)) \
        +(~wormhole_condition).astype(int)*self.pop_mat
        
        self.pop_mat=np.clip(self.pop_mat,a_min=self.lb,a_max=self.ub)
        
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
        
        fig.suptitle("Multi-Verse Optimization", fontsize=20, fontweight=title_weight)
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
        mp.xlabel("Number of Generation",fontsize=axis_font, fontweight=axis_weight)
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

    def f(x1,x2,x3):
        y=(x1-50)**2+(x2-20)**2+(x3-100)**2
        return y
    
    #(self,f,x,lb,ub,pop=200,max_gen=50,min=0.2,max=1,p=6,verbose=True):
    mvo=MultiVerseOptimizer(f,["x1","x2","x3"],[0,0,5],[100,50,100],pop=200,max_gen=500,min=0.2,max=1,p=6,verbose=True)
    mvo.solve()
    mvo.plot_result()   
    
       
        

