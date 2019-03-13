#%%
def copy_paste(infile):
    infilename = 'raw/'+infile
    a = open(infilename)
    x = a.readlines()
    x = x[2:]
    newname = 'clean/'+infile
    b = open(newname,'w')
    b.writelines(x)
    #%%
    for i in range(64):
        for j in range(31):
           infile = str(i+1)+'-'+str(j)+'.xyz'
           copy_paste(infile)
#%%