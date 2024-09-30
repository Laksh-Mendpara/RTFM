import visdom
import torch
import torch.nn
#python -m visdom.server


class Visualizer(object):
    def __init__(self,env="default",**kwargs): 
        #whatever extra keywords added in the init method of the visdom is passed from here
        self.vis=visdom.Visdom(env=env,**kwargs)
        self.index={}
        
    def plot_lines(self,name,y,**kwargs):
        #x=self.index.get(name, 1.00)
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name), #displays a name on image
                      opts=dict(title=name), 
                      update=None if x==0 else 'append',
                      **kwargs
                      )
        self.index[name]=x+1
        
    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))
        
    def lines(self, name, line, X=None):
        if X is None: #if there is no x axis specified then it takes index as x
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)
            
    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)
      
    #extra faeture added  
    def plot_confusion_matrix(self, name, cm, labels):
        self.vis.heatmap(X=cm, win=name, opts=dict(
        title=name,
        rownames=labels,
        columnnames=labels
    ))
    
    def save_visualizations(self, path):
        self.vis.save([path])
        #yaha path add karna hai
        
    
    def minmax_norm(act_map, min_val=None, max_val=None):
        if min_val is None or max_val is None: #if min/max value is not specified then it finds it, else it is skipped directly to delta part...
            relu=torch.nn.ReLU()
            max_val=relu(torch.max(act_map, dim=0)[0])
            min_val=relu(torch.min(act_map, dim=0)[0])
        delta=max_val-min_val
        delta[delta<=0]=1
        ret=(act_map-min_val)/delta #value belongs to range [0,1]
        ret[ret>1]=1
        ret[ret<0]=0
        return ret


        