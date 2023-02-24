class Encoder(nn.Module):
    '''
    Encodes gene expression vectors to latent space of dim hidden_dims[-1]
    '''
    def __init__(self, input_dim, hidden_dims):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[0], hidden_dims[1])
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class Decoder(nn.Module):
    '''
    Decodes gene expression latent vectors back to dim output_dim
    '''
    def __init__(self, output_dim, hidden_dims):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[0], output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class PertEncoder(nn.Module):
    '''
    We now want to add in the effect of the perturbations on the basal state. 
    Almost analogous to sDMD, we will do this using a nonlinear function and assume 
    that the perturbation impact is linear in the latent space. 
    '''
    def __init__(self, pert_dim, basal_dim):
        super(PertEncoder, self).__init__()
        
        self.fc1 = nn.Linear(pert_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, basal_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class PertDecoder(nn.Module):
    '''
    Map the encoded perturbations back to the original space. This allows us to 
    interpolate in perturbation latent space and then generate new hypotheses. 
    '''
    def __init__(self, pert_dim, basal_dim):
        super(PertDecoder, self).__init__()
        
        self.fc1 = nn.Linear(basal_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, pert_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # want positive values
        return x
    
    
class Autoencoder(nn.Module):
    '''
    Gene expression autoencoder + perturbation in latent space. 
    '''
    
    def __init__(self, input_dim, hidden_dims, pert_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims)
        self.decoder = Decoder(input_dim, hidden_dims)
        self.pertencoder = PertEncoder(pert_dim, hidden_dims[-1])
        
    def forward(self, x, p): # p is the vector of perturbations
        x = self.encoder(x)
        x = x + self.pertencoder(p)
        x = self.decoder(x)
        return x
    
    
class Discriminator(nn.Module):
    '''
    Takes in basal state and uses it to predict the continuous perturbation vector. 
    The goal is to have the discriminator do no better than guessing, meaning that the 
    basal state contains little to no information about the perturbations. 
    '''
    def __init__(self, input_dim, pert_dim):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, pert_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        