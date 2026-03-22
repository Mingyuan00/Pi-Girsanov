import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.utils.data.dataset import random_split

from deeptime.decomposition import TICA
from deeptime.covariance import KoopmanWeightingEstimator

import numpy as np

#### FMRC Functions
def run_TICA(data,lagtime,dim=None,var_cutoff=None,koopman=True):
    tica = TICA(lagtime=lagtime,dim=dim,var_cutoff=var_cutoff)
    if koopman == True:
        koopman_estimator = KoopmanWeightingEstimator(lagtime=lagtime)
        reweighting_model = koopman_estimator.fit(data).fetch_model()
        tica = tica.fit(data, weights=reweighting_model).fetch_model()
    else:
        tica = tica.fit(data).fetch_model()
    # tica is the data-fitted model, which contains eigenvalues and eigenvectors
    # tica_output is the tranformed time-series data in TICA space in shape(traj_idx,no_frames,dim)
    # tica_output_concat is tica_output in shape(traj_idx*no_frames,dim)
    tica_output = tica.transform(data)
    tica_output_concat = np.concatenate(tica_output)
        
    return tica,tica_output,tica_output_concat

def create_timelagged_dataset(tica_output,lagtime):
    timelagged_dataset = []
    tica_data = list(tica_output)
    for tica_data_i in tica_data:
        for i in range(tica_data_i.shape[0]-lagtime):
            lagged_pair_i = np.vstack([tica_data_i[i],tica_data_i[i+lagtime]])
            lagged_pair_i = np.array(lagged_pair_i)
            timelagged_dataset.append(lagged_pair_i)
    timelagged_dataset = torch.tensor(np.array(timelagged_dataset),dtype=torch.float64)
    return timelagged_dataset

def minmax_normalization(data,axis=0):
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

# FMRC
class GaussianPrior(nn.Module):
    def __init__(self, mean_value, std_value):
        super().__init__()
        self.mean_ = nn.Parameter(torch.tensor(mean_value, dtype=torch.float64), requires_grad=False)
        self.std_ = nn.Parameter(torch.tensor(std_value, dtype=torch.float64), requires_grad=False)

    def forward(self, size):
        samples = self.mean_ + self.std_ * torch.randn(size).to(self.mean_.device)
        return samples

    def sample_like(self, x):
        size = x.size()
        samples = self.forward(size).to(x.device)
        return samples


class FMRC(nn.Module):
    def __init__(self,input_size,latent_size,hidden_size,hidden_depth,activation,
                 sigma,learning_rate,lr_decay,lr_decay_stepsize,val_frac,batch_size,n_epochs,device):
        super().__init__()
        
        # Neural network related
        self.input_size = input_size               # No. of features
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.activation = activation
        self.sigma = sigma                         # the gaussian width of flow matching vector field sample, 
                                                   # serves as a regularization factor

        # Training related
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_stepsize = lr_decay_stepsize
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

        # Cached attributes
        self.encoder = None
        self.L_vector_field = None
        self.D_vector_field = None
        self.optimizer = None
        self.scheduler = None
        self.train_loss = None
        self.validation_loss = None
        
        #####
        
        ### Encoder x --> r(x) as a feed forward nn
        self.encoder = []
        
        # first layer
        self.encoder.append(nn.Linear(self.input_size,self.hidden_size,dtype=torch.float64))
        self.encoder.append(self.activation)
        # middle layers
        for i in range(self.hidden_depth-1):
            self.encoder.append(nn.Linear(self.hidden_size,self.hidden_size,dtype=torch.float64))
            self.encoder.append(self.activation)
        # final output layer
        self.encoder.append(nn.Linear(self.hidden_size,self.latent_size,dtype=torch.float64))
        self.encoder = nn.Sequential(*self.encoder).to(self.device)

        ### nn that represents the lumpability vector field u(r(x),y,t)
        self.L_vector_field = []
        L_input_size = self.latent_size + self.input_size + 1
        
        # first layer
        self.L_vector_field.append(nn.Linear(L_input_size,self.hidden_size,dtype=torch.float64))
        self.L_vector_field.append(self.activation)
        # middle layers
        for i in range(self.hidden_depth-1):
            self.L_vector_field.append(nn.Linear(self.hidden_size,self.hidden_size,dtype=torch.float64))
            self.L_vector_field.append(self.activation)
        # final output layer
        self.L_vector_field.append(nn.Linear(self.hidden_size,self.input_size,dtype=torch.float64))
        self.L_vector_field = nn.Sequential(*self.L_vector_field).to(self.device)

        #####
        
        ### nn that represents the decomposibility vector field u(r(y),x,t)
        self.D_vector_field = []
        D_input_size = self.latent_size + self.input_size + 1
        
        # first layer
        self.D_vector_field.append(nn.Linear(D_input_size,self.hidden_size,dtype=torch.float64))
        self.D_vector_field.append(self.activation)
        # middle layers
        for i in range(self.hidden_depth-1):
            self.D_vector_field.append(nn.Linear(self.hidden_size,self.hidden_size,dtype=torch.float64))
            self.D_vector_field.append(self.activation)
        # final output layer
        self.D_vector_field.append(nn.Linear(self.hidden_size,self.input_size,dtype=torch.float64))
        self.D_vector_field = nn.Sequential(*self.D_vector_field).to(self.device)

        #####
        
    def encode(self,x):
        # Here, x is the variable to be encoded i.e. for lumpability x:= x for decomposibility x:= y
        # encode x --> r:=r(x) & scale(L2_normalize(r(x)))
        r = self.encoder(x)
        return r

    def sample_from_prior(self,x):
        # Sample x/y (D_loss/L_loss) from prior, prior_sample should have shape (batch_size,input_size)
        # Since this is just a gaussian with mean & s.t.d from all data, x/y can share the same prior
        prior_sample = self.prior.sample_like(x)
        return prior_sample

    def sample_t(self,x):
        # Sample t = [t_1,...t_B], t should have shape (batch_size,)
        t = torch.rand_like(x[:,:1])
        return t

    def sample_x_t(self,x,t,prior_sample): 
        # Sample x_t/y_t, the 'location' of x/y after 'time' t in the vector field
        x_t = x * t + (1-t) * prior_sample + torch.randn_like(x) * self.sigma
        return x_t

    def data_vector_field(self,x,prior_sample):
        # Calculate v_t, the 'data vector field' that we want to match our 'neural network vector field' with
        v_t = x - prior_sample
        return v_t

    def L_loss(self,x,y,t):
        # Step 1: encode x --> r(x)
        rx = self.encode(x)
        
        # Step 2: sample y from prior
        prior_y = self.sample_from_prior(y)
        
        # Step 3: sample y_t, y at time t in the vector field
        y_t = self.sample_x_t(y,t,prior_y)
        
        # Step 4: compute 'data vector field'
        v_t = self.data_vector_field(y,prior_y)
        
        # Step 5: compute 'nn vector field'
        u_input = torch.cat((rx,y_t,t),dim=-1)
        # so that u_t is a function of rx,y_t,t only i.e. rx contains nearly same information as x
        u_t = self.L_vector_field(u_input)
        
        # Step 6: compute flow matching loss
        L_loss = torch.mean(torch.sum((u_t-v_t)**2,-1))
        
        return L_loss
    
    def D_loss(self,x,y,t):
        # Step 1: encode y --> r(y)
        ry = self.encode(y)
        
        # Step 2: sample x from prior
        prior_x = self.sample_from_prior(x)
        
        # Step 3: sample x_t, x at time t in the vector field
        x_t = self.sample_x_t(x,t,prior_x)
        
        # Step 4: compute 'data vector field'
        v_t = self.data_vector_field(x,prior_x)
        
        # Step 5: compute 'nn vector field'
        u_input = torch.cat((ry,x_t,t),dim=-1)
        # so that u_t is a function of rx,y_t,t only i.e. rx contains nearly same information as x
        u_t = self.L_vector_field(u_input)
        
        # Step 6: compute flow matching loss
        D_loss = torch.mean(torch.sum((u_t-v_t)**2,-1))
        
        return D_loss

    def fit(self,data,lagtime):
        # NB: usually we use tica_output as data
        # Initialize the gaussian prior
        prior_mean = np.mean(np.concatenate(data),axis=0)
        prior_sigma = np.std(np.concatenate(data),axis=0)
        self.prior = GaussianPrior(prior_mean,prior_sigma).to(self.device)
        
        # Create time-lagged dataset: this outputs a 3d numpy array with shape (no_frame,2,no_features)
        # the second dimension represents the time-lagged pairs X_t,X_t+tau
        dataset = create_timelagged_dataset(data,lagtime)
        
        # Create training set and validation set
        n_pairs = len(dataset)
        train_size = int((1-self.val_frac)*n_pairs)
        val_size = n_pairs - train_size
        train_data, val_data = random_split(dataset,[train_size,val_size])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.batch_size,shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = self.batch_size)  
        
        # Training
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_stepsize, gamma=self.lr_decay)
        
        train_loss = []
        validation_loss = []

        # Create a separate training set and validation set for topology loss
        data_concat = torch.tensor(np.concatenate(data,axis=0),dtype=torch.float64)
        
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(1, self.n_epochs + 1):
                train_loss_epoch = []
                validation_loss_epoch = []
                    
                # Training
                for i,minibatch_data in enumerate(train_loader):
                    # Prepare x and y, should both in shape (batchsize,no_features)
                    x = minibatch_data[:,0,:].to(self.device)
                    y = minibatch_data[:,1,:].to(self.device)
                    # Sample t
                    t = self.sample_t(x)
                    # Compute losses
                    L_loss = self.L_loss(x,y,t)
                    D_loss = self.D_loss(x,y,t)
                    loss = L_loss + D_loss
                    # back propagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # Record current minibatch loss
                    train_loss_epoch.append(loss.item())

                train_loss_epoch = np.mean(train_loss_epoch)
                train_loss.append(train_loss_epoch)
                
                # learning rate decay
                self.scheduler.step()

                # Validation
                with torch.no_grad():
                    for i,minibatch_data in enumerate(val_loader):
                        # Prepare x and y, should both in shape (batchsize,no_features)
                        x = minibatch_data[:,0,:].to(self.device)
                        y = minibatch_data[:,1,:].to(self.device)
                        # Sample t
                        t = self.sample_t(x)
                        # Compute losses
                        L_loss = self.L_loss(x,y,t)
                        D_loss = self.D_loss(x,y,t)
                        loss = L_loss + D_loss
                        # Record current minibatch loss
                        validation_loss_epoch.append(loss.item())
                    validation_loss_epoch = np.mean(validation_loss_epoch)
                    validation_loss.append(validation_loss_epoch)

                print('Epoch {}: Total Train loss = {:.4f}, validation loss = {:.4f}'.format(epoch,train_loss_epoch,validation_loss_epoch))

        self.train_loss = train_loss
        self.validation_loss = validation_loss
        return None
        
    def transform(self,data_concat,batchsize=None):
        if batchsize == None:
            batchsize = self.batch_size
        r = []
        no_iteration = data_concat.shape[0]//batchsize + 1
        for i in range(no_iteration):
            with torch.no_grad():
                r_i = self.encode(torch.tensor(data_concat[i*batchsize:(i+1)*batchsize],dtype=torch.float64).to(self.device))
                r.append(r_i.cpu().detach().numpy())
                del r_i
                torch.cuda.empty_cache
        r = np.concatenate(r,axis=0)
        return r

    def save_model(self,filepath):
        torch.save(self,filepath)
        return None
