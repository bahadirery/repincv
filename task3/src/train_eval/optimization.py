import torch.optim as optim

def lrdecay_scheduler(optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every 10 epochs. Add the paper reference here."""
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    return scheduler

def cosineannealing(config_params, optimizer):
    
    scheduler =optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_params['maxepochs'])
    
    return scheduler

def select_lr_scheduler(config_params, optimizer):
    if config_params['trainingmethod'] == 'lrdecay':
        scheduler = lrdecay_scheduler(optimizer)
    if config_params['trainingmethod'] == 'cosine':
        scheduler = cosineannealing(config_params, optimizer)
    else:
        scheduler = None
    return scheduler

def optimizer_fn(config_params, model):
    classifier=[]
    rest_group=[]
    if config_params['optimizer_fn']=='Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'], weight_decay=config_params['wtdecay'])
        
    elif config_params['optimizer_fn']=='SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'], momentum=0.9, weight_decay=config_params['wtdecay'])
    return optimizer