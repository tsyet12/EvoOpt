from solver import MGoldenSearch as MGS
		
if __name__=="__main__":
	def f(a):
		return (a+5)*(a+5)
	
	def g(a,b):
		return (a+5)*(a+5)+(b+1)*(b+1)
	
	opt=MGS.MGoldenSearch(f,[-10],[10])
	result1=opt.solve()
	#print(result1)
	
	opt2=MGS.MGoldenSearch(g,[-10,-5],[10,5])
	result2=opt.solve()
	print(result2)
	