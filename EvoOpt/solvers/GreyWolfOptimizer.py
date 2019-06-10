'''Grey Wolf Optimizer
###
###Code and Implementation by Sin Yong, Teng
###
###
###Implemented on 09/06/2019
'''


import numpy as np
import matplotlib.pyplot as mp

class GreyWolfOptimizer():
    def __init__(self,f,x,lb,ub,pop=200,max_gen=50,a_max=2,a_min=0,verbose=True):
        self.f=np.vectorize(f)
        self.x=x
        self.lb=lb
        self.ub=ub
        self.pop=pop
        self.verbose=verbose
            
        self.max_gen=max_gen
        self.pop_mat=np.tile(self.lb,(pop,1))+np.random.rand(pop,len(x)).astype(np.longdouble)*(np.tile(self.ub,(pop,1))-np.tile(self.lb,(pop,1)))
        
        self.a=a_max-(a_max-a_min)/(self.max_gen)*np.arange(self.max_gen+1)
        self.good_wolf=3
        
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
        
        #get prey position
        self.xp=np.clip(np.average(self.pop_mat[:self.good_wolf,:],axis=0),self.lb,self.ub)
       
    def update(self,generation):
        vec_a=np.full(shape=(len(self.x),1),fill_value=self.a[generation])
        r1=np.random.rand(*vec_a.shape)
        r2=np.random.rand(*vec_a.shape)
        A=np.tile(2*vec_a*r1-vec_a,self.pop-self.good_wolf).reshape((len(self.x),self.pop-self.good_wolf)).T
        C=2*r2
        D=np.abs(np.tile(C,self.pop-self.good_wolf).reshape((len(self.x),self.pop-self.good_wolf)).T*\
            np.tile(self.xp,self.pop-self.good_wolf).reshape((len(self.x),self.pop-self.good_wolf)).T \
            -self.pop_mat[self.good_wolf:,:])
        
        self.pop_mat[self.good_wolf:,:]=self.pop_mat[self.good_wolf:,:]-A*D
        
        r1_s=np.random.rand(*self.pop_mat[:self.good_wolf,:].shape)
        r2_s=np.random.rand(*self.pop_mat[:self.good_wolf,:].shape)        
        
        D_s=np.abs((2*r2_s)*self.pop_mat[:self.good_wolf,:]-np.tile(self.xp,self.good_wolf).reshape((len(self.x),self.good_wolf)).T)
        self.pop_mat[:self.good_wolf,:]=self.pop_mat[:self.good_wolf,:]-(2*self.a[generation]*r1_s-self.a[generation])*D_s
        
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
        try:
            mng = mp.get_current_fig_manager()
            mng.window.state('zoomed')
        except:
            print("Format your plot using: matplotlib.rcParams['figure.figsize'] = [width, height]")
        mp.show()

if __name__=="__main__":

    def f(x1,x2,x3):
        y=(x1-50)**2+(x2-20)**2+(x3-100)**2
        return y
    
    #(self,f,x,lb,ub,pop=200,max_gen=50,min=0.2,max=1,p=6,verbose=True):
    gwo=GreyWolfOptimizer(f,["x1","x2","x3"],[0,0,5],[100,50,100],pop=200,max_gen=100,a_max=2,a_min=0,verbose=True)
    gwo.solve()
    gwo.plot_result()   
    
       
        

