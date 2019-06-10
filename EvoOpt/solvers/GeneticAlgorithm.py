'''Genetic Algorithm
###
###Code and Implementation by Sin Yong, Teng
###
###
###Implemented on 31/05/2019
'''


import numpy as np
import matplotlib.pyplot as mp

class GeneticAlgorithm():
    def __init__(self,f,x,lb,ub,pop=200, max_gen=50,mut_prob=0.05,cross_prob=0.5,num_elite=5,verbose=True,roll_cross=0):
        self.f=np.vectorize(f)
        self.x=x
        self.lb=lb
        self.ub=ub
        self.pop=pop
        self.max_gen=max_gen
        self.pop_mat=np.tile(self.lb,(pop,1))+np.random.rand(pop,len(x)).astype(np.longdouble)*(np.tile(self.ub,(pop,1))-np.tile(self.lb,(pop,1)))
        self.plotgen=[]
        self.average_fit=[]
        self.cross_prob=cross_prob
        self.mut_prob=mut_prob
        self.num_elite=num_elite
        self.verbose=verbose
        self.roll_cross=roll_cross
        self.history1=self.pop_mat[:,0]
        self.history2=self.pop_mat[:,1]
        
        #initialize elitist population if required
        if self.num_elite>0:
            random_elite=np.tile(self.lb,(num_elite,1))+np.random.rand(num_elite,len(x)).astype(np.longdouble)*(np.tile(self.ub,(num_elite,1))-np.tile(self.lb,(num_elite,1)))
            self.elite_mat=np.concatenate((self.f(*random_elite.T).reshape(self.num_elite,1),random_elite),axis=1)
        self.best_result=[]
        self.best_domain=[]
        self.overall_best=[]
        self.overall_bestdomain=[]
        
    def solve(self):
        self.evaluate_fitness()
        
        for i in range(self.max_gen+1):
            self.rand_selection()
            
            self.crossover_mutate()
            self.evaluate_fitness()
            if self.verbose:
                self.log_result(i)
            
    
    def evaluate_fitness(self):
    
        #put elitist into population for fitness evaluation
        if self.num_elite>0:
            self.pop_mat=np.concatenate((self.elite_mat[:,1:],self.pop_mat),axis=0)
        #get fitness of all population
        self.pop_mat_fit=self.f(*self.pop_mat.T)
        #concatenate and sort population by fitness
        temp_mat=np.concatenate((np.asarray(self.pop_mat_fit).reshape(self.pop_mat_fit.shape[0],1),self.pop_mat),axis=1)
        temp_mat=temp_mat[temp_mat[:,0].argsort()]
        
        #store elite if elitist GA is enabled
        if self.num_elite>0:
            self.elite_mat=np.concatenate((self.elite_mat,temp_mat[:self.num_elite,:]),axis=0)
            self.elite_mat=self.elite_mat[self.elite_mat[:,0].argsort()][:self.num_elite,:]
            
        #print(temp_mat)
        #split fitness and population
        self.pop_mat_fit, self.pop_mat =temp_mat[:,0], temp_mat[:,1:]
        
        #save the evaluated points
        self.history1=np.concatenate((self.history1,self.pop_mat[:,0]),axis=0)
        self.history2=np.concatenate((self.history2,self.pop_mat[:,1]),axis=0)
       
    def rand_selection(self):
        #selection of good genes up to population size
        self.pop_mat=self.pop_mat[:self.pop,:]
        
        #crease a random shuffled mirror duplicate for crossover and mutation
        self.pop_rand_dup=np.copy(self.pop_mat)
        
        if self.roll_cross==0:
            #if roll crossover is not enabled, use random cross
            np.random.shuffle(self.pop_rand_dup)
        else:
            #other wise roll the crossover duplicate according to roll number
            np.roll(self.pop_rand_dup,self.roll_cross,axis=0)
        #self.pop_rand_dup=self.pop_rand_dup        
        
    def crossover_mutate(self):
        #create a additional population matrix to store offspring genes
        self.add_pop=np.empty_like(self.pop_mat)
        #randomize cross index to compare with cross probability
        self.rcross_prob=np.random.rand(*self.pop_mat.shape)
        #randomize mutation index to compare with mutation probability
        self.rmut_prob=np.random.rand(*self.pop_mat.shape)
        #pre-randomize a matrix for mutation
        mut_matrix=np.tile(self.lb,(self.pop_mat.shape[0],1))+np.random.rand(self.pop_mat.shape[0],len(self.x)).astype(np.longdouble)*(np.tile(self.ub,(self.pop_mat.shape[0],1))-np.tile(self.lb,(self.pop_mat.shape[0],1)))
        #selection and mutation
        for i in range(self.rcross_prob.shape[0]):
            for j in range(self.rcross_prob.shape[1]):
                if self.rmut_prob[i][j]<=self.mut_prob:
                    self.add_pop[i][j]=mut_matrix[i][j] # first look if gene is mutated. Priority of mutation is higher than crossover.
                elif self.rcross_prob[i][j]<=self.cross_prob:
                    self.add_pop[i][j]=self.pop_rand_dup[i][j] #second look if gene is crossovered
                else:
                    self.add_pop[i][j]=self.pop_mat[i][j] #if both do not happen,then take gene from original pop
                    
        #print("pop mat",self.pop_mat)
        #print("pop_rand_dup",self.pop_rand_dup)
        #print("add pop",self.add_pop)
        
        
        self.pop_mat=np.concatenate((self.pop_mat,self.add_pop),axis=0)
        
        
        
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
            
            
            
            
            

    def plot_result(self,contour_density=50):
        subtitle_font=16
        axis_font=14
        title_weight="bold"
        axis_weight="bold"
        tick_font=14
		
		
        fig=mp.figure()
        
        fig.suptitle("Genetic Algorithm Optimization", fontsize=20, fontweight=title_weight)
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
        y=x1+x3+x2*x2
        return y
    
    ga=GeneticAlgorithm(f,["x1","x2","x3"],[0,0,5],[100,50,100],pop=200,max_gen=50,mut_prob=0.05,cross_prob=0.5,num_elite=2,verbose=True, roll_cross=-1)
    ga.solve()
    ga.plot_result()   
    
       
        

