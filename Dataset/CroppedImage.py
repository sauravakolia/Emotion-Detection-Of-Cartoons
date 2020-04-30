import pandas as pd



file1 = open('textfiles\\''ftestresult.txt', 'r') 

i=0
my_file=[]
Lines = file1.readlines() 
for line in Lines: 
    s=line.strip()
    if(len(s)>=18 and s[18]=='C'):
    	i=i+1
    my_file.append(s)

 
print(i)
d={'Location':str,
'Tom':int,
'Jerry':int,
'TL':int,
'TT':int,
'TW':int,
'TH':int,
'JL':int,
'JT':int,
'JW':int,
'JH':int}
df = pd.DataFrame(index=range(i),columns=['Location','Tom','Jerry','TL','TT','TW','TH','JL','JT','JW','JH'])
for col in df.columns:
    df[col].values[:] = 0

df=df.astype(d)    

print(df.dtypes)
# for col in df.columns:
#     df[col].values[:] = 0


# print(df)

# new_file=[]

# df = pd.DataFrame(['Location','Tom','Jerry','TL','TT','TW','TH','JL','JT','JW','JH']) 

# # df.loc[df.index[someRowNumber], 'New Column Title'] = "some value"
# dfObj.loc['b'] = [24, 'Aadi', 'Logout']

k=-1
c=0
for x in my_file:
	x=x.split()

	if(k<=i-2 and x[0]=='Enter'):
		k=k+1
		df.loc[df.index[k],'Location']=x[3]+' '+x[4][0:-1]
		
		
		

	if(x[0]=='Tom:'):
		
		df.loc[df.index[k],'Tom']=1
		df.loc[df.index[k],'TL']=int(x[3])
		df.loc[df.index[k],'TT']=int(x[5])
		df.loc[df.index[k],'TW']=int(x[7])
		df.loc[df.index[k],'TH']=int((x[9])[0:-1])
		# df.loc[k]['TL']=x[3]
		# df.loc[k]['TT']=x[5]
		# df.loc[k]['TW']=x[7]
		# df.loc[k]['TH']=x[9]	
		# file=[]
		# file.append(x[19:])

	if(x[0]=='Jerry:'):
		df.loc[df.index[k],'Jerry']=1
		df.loc[df.index[k],'JL']=int(x[3])
		df.loc[df.index[k],'JT']=int(x[5])
		df.loc[df.index[k],'JW']=int(x[7])
		df.loc[df.index[k],'JH']=int((x[9])[0:-1])
		# df.loc[k]['JL']=x[3]
		# df.loc[k]['JT']=x[5]
		# df.loc[k]['JW']=x[7]
		# df.loc[k]['JH']=x[9]	

		
df.loc[df.index[k],'Location']=my_file[-3].split()[3]+' '+my_file[-3].split()[4]


print(df)
print(df.dtypes)
df.to_csv('testresult.csv')