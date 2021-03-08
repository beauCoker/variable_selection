# standard library imports
import os
import math

# package imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Callback:
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def on_train_begin(self, n_epochs, metrics):
        # overwrite in subclass
        pass

    def on_epoch_end(self, epoch, history):
        # overwrite in subclass
        # return True if training should end
        pass

    def on_train_end(self, epoch, history):
        # overwrite in subclass
        pass

class CallbackList:
    def __init__(self, callback_list, model, optimizer):
        self.callback_list = callback_list
        [c.set_model(model) for c in self.callback_list]
        [c.set_optimizer(optimizer) for c in self.callback_list]

    def on_train_begin(self, n_epochs, metrics):
        [c.on_train_begin(n_epochs, metrics) for c in self.callback_list]

    def on_epoch_end(self, epoch, history):
        return [c.on_epoch_end(epoch, history) for c in self.callback_list]

    def on_train_end(self, epoch, history):
        [c.on_train_end(epoch, history) for c in self.callback_list]


class EarlyStopper(Callback):
    def __init__(self, frac_begin_lookingback=.75, frac_lookback=.25, improvement_threshold=0.0):
        super(EarlyStopper).__init__()
        self.frac_begin_lookingback = frac_begin_lookingback
        self.frac_lookback = frac_lookback
        self.improvement_threshold = improvement_threshold

    def on_train_begin(self, n_epochs, metrics=None):
        self.begin_lookingback = round(self.frac_begin_lookingback*n_epochs)
        n_epochs_lookback = round(self.frac_lookback*n_epochs)

        self.lookback_start = self.begin_lookingback - n_epochs_lookback

        self.loss_best_before = None
        self.loss_best_lookback = None

    def on_epoch_end(self, epoch, history=None):
        # note: recomputes best loss each time, could be improved
        if epoch >= self.begin_lookingback:
            
            '''
            if self.loss_best_before is None:
                self.loss_best_before = min(history['loss'][:self.lookback_start]) # first time look at all
            else:
                self.loss_best_before = min((self.loss_best_before, history['loss'][self.lookback_start-1]))
            '''
            self.loss_best_before = min(history['loss'][:self.lookback_start])
            self.loss_best_lookback = min(history['loss'][-self.lookback_start:])

            percent_improvement = (self.loss_best_before - self.loss_best_lookback)/abs(self.loss_best_before) # positive is better
            if percent_improvement <= self.improvement_threshold:
                print('stopping early at epoch = %d' % epoch)
                return True
            else:
                self.lookback_start += 1


class Saver(Callback):
    def __init__(self, frac_start_save=.9, dir_checkpoint='./'):
        super(Saver).__init__()
        self.frac_start_save = frac_start_save
        self.dir_checkpoint = dir_checkpoint

    def on_train_begin(self, n_epochs, metrics=None):
        self.epoch_start_save = round(self.frac_start_save*n_epochs)
        self.loss_best_saved = torch.tensor(float('inf'))
        self.saved_model = False

        if self.frac_start_save < 1:
            if not os.path.exists(self.dir_checkpoint):
                os.makedirs(self.dir_checkpoint)

    def on_epoch_end(self, epoch, history):
        loss = history['loss'][epoch]
        if epoch >= self.epoch_start_save and loss < self.loss_best_saved:
            print('saving mode at epoch = %d' % epoch)
            self.saved_model = True
            self.loss_best_saved = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            },  os.path.join(self.dir_checkpoint, 'checkpoint.tar'))

    def on_train_end(self, epoch, history):
        # reload best model if saving
        loss = history['loss'][epoch]
        if self.saved_model and self.loss_best_saved < loss:
            checkpoint = torch.load(os.path.join(self.dir_checkpoint, 'checkpoint.tar'))
            print('reloading best model from epoch = %d' % checkpoint['epoch'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

        else:
        	# save final model
        	torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            },  os.path.join(self.dir_checkpoint, 'checkpoint.tar'))
        

class Printer(Callback):
    def __init__(self, frac_print=.1):
        super(Saver).__init__()
        self.frac_print = frac_print

    def on_train_begin(self, n_epochs, metrics):
        print('initial state:')
        self.n_epochs = n_epochs
        self.n_epochs_print = max((1,round(self.frac_print*n_epochs)))
        self.print(0, metrics)

    def on_epoch_end(self, epoch, history):
        if epoch % self.n_epochs_print==0:
            metrics = self.extract_metrics(epoch, history)
            self.print(epoch, metrics)

    def on_train_end(self, epoch, history):
        print('final state:')
        metrics = self.extract_metrics(epoch, history)
        self.print(epoch, metrics)

    def extract_metrics(self, epoch, history):
        metrics = {}
        for key, value in history.items():
            metrics[key]=value[epoch] 
        return metrics

    def print(self, epoch, metrics):
        out = 'epoch [%d/%d]: ' % (epoch, self.n_epochs)
        for item in metrics.items():
            out += '%s: %.3f, ' % item
        print(out)


class Temperer(Callback):
    '''
    For use with models that have a set_temperature method
    '''
    def __init__(self, frac_stop_temper=.1, temperature_min=0.0, temperature_max=1.0):
        super(Temperer).__init__()
        self.frac_stop_temper = frac_stop_temper
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

    def on_train_begin(self, n_epochs, metrics=None):
        self.epoch_stop_temper = round(self.frac_stop_temper*n_epochs)

    def on_epoch_end(self, epoch, history=None):
        if epoch <= self.epoch_stop_temper:
            temperature = self.temperature_min*(epoch/self.epoch_stop_temper)
        else:
            temperature = self.temperature_max
        self.model.set_temperature(temperature)


class SGDRScheduler(Callback):
    '''
    Adapted from: https://www.jeremyjordan.me/nn-learning-rate/

    min_lr: The lower bound of the learning rate range for the experiment.
    max_lr: The upper bound of the learning rate range for the experiment.
    steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
    lr_decay: Reduce the max_lr after the completion of each cycle.
              Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
    cycle_length: Initial number of epochs in a cycle.
    mult_factor: Scale epochs_to_restart after each full cycle completion.

    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):
        super(SGDRScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self):
        pass

    def on_epoch_end(self, epoch, metrics):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            #self.best_weights = self.model.get_weights() # add this functionality later? 

    def on_train_end(self, epoch, metrics):
        pass
        #self.model.set_weights(self.best_weights) # add this functionality later? 

class SGDRScheduler(Callback):
    '''
    Adapted from: https://www.jeremyjordan.me/nn-learning-rate/

    min_lr: The lower bound of the learning rate range for the experiment.
    max_lr: The upper bound of the learning rate range for the experiment.
    steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
    lr_decay: Reduce the max_lr after the completion of each cycle.
              Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
    cycle_length: Initial number of epochs in a cycle.
    mult_factor: Scale epochs_to_restart after each full cycle completion.

    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):
        super(SGDRScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.epoch_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.epoch_since_restart / self.cycle_length
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self):
        pass

    def on_epoch_end(self, epoch, metrics):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.epoch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
        else:
            self.optimizer.lr = self.clr() # this won't work
            self.epoch_since_restart += 1

    def on_train_end(self, epoch, metrics):
        pass
        


class WeightsRecorder(Callback):
    def __init__(self):
        super(WeightsRecorder).__init__()

        self.transform = nn.Softplus()
    
    def on_train_begin(self, n_epochs, metrics):
        
        # parameter indices
        pnames = [name for name, param in self.model.named_parameters()]
        self.idx_means = [i for (i, pname) in enumerate(pnames) if 'loc' in pname]
        self.idx_stds = [i for (i, pname) in enumerate(pnames) if 'scale' in pname]
        
        # initial parameters
        means, stds = self.flatten_params()
        self.means_init = means
        self.stds_init = stds
        
        self.means_init_norm = torch.sqrt(torch.sum(self.means_init**2))
        self.stds_init_norm = torch.sqrt(torch.sum(self.stds_init**2))
        
        # allocate space
        self.means_rel_change = []
        self.stds_rel_change = []

    def on_epoch_end(self, epoch, history):
        means, stds = self.flatten_params()
        
        self.means_rel_change.append(self.rel_change(means, self.means_init, self.means_init_norm))
        self.stds_rel_change.append(self.rel_change(stds, self.stds_init, self.stds_init_norm))
    
    def rel_change(self, x, x0, x0_norm):
        return torch.sqrt(torch.sum((x-x0)**2))/x0_norm
        
    def flatten_params(self):
        parameters = [p.detach().clone().reshape(-1) for p in self.model.parameters()]    
        means = torch.cat([parameters[i] for i in self.idx_means])        
        #stds = torch.exp(torch.cat([parameters[i] for i in self.idx_stds])) # oops! shouldn't be exponentiating
        stds = self.transform(torch.cat([parameters[i] for i in self.idx_stds])) 
        
        return means, stds



class KLGradRecorder(Callback):
    def __init__(self):
        super(KLGradRecorder).__init__()

        #self.transform = nn.Softplus()
    
    def on_train_begin(self, n_epochs, metrics):
        
        # parameter indices
        pnames = [name for name, param in self.model.named_parameters()]
        self.idx_means = [i for (i, pname) in enumerate(pnames) if 'loc' in pname]
        self.idx_stds = [i for (i, pname) in enumerate(pnames) if 'scale' in pname]
        
        # initial parameters
        means, stds = self.flatten_grads()
        self.means_init = means
        self.stds_init = stds
        
        self.means_init_norm = torch.sqrt(torch.sum(self.means_init**2))
        self.stds_init_norm = torch.sqrt(torch.sum(self.stds_init**2))
        
        # allocate space
        self.means_rel_change = []
        self.stds_rel_change = []

    def on_epoch_end(self, epoch, history):
        means, stds = self.flatten_grads()

        self.means_rel_change.append(self.rel_change(means, self.means_init, self.means_init_norm))
        self.stds_rel_change.append(self.rel_change(stds, self.stds_init, self.stds_init_norm))
    
    def rel_change(self, x, x0, x0_norm):
        return torch.sqrt(torch.sum((x-x0)**2))/x0_norm
        
    def flatten_grads(self):
        '''
        Computes KL and returns gradients
        '''
        with torch.enable_grad():
            self.optimizer.zero_grad()
            kl = self.model.kl_divergence()
            kl.backward()

        grads = [p.grad.detach().clone().reshape(-1) for p in self.model.parameters()]    
        means = torch.cat([grads[i] for i in self.idx_means])        
        stds = torch.cat([grads[i] for i in self.idx_stds]) # untransformed
        
        '''
        ###
        parameters = [p.detach().clone().reshape(-1) for p in self.model.parameters()]   
        means_param = torch.cat([parameters[i] for i in self.idx_means]) 
        stds_param = torch.cat([parameters[i] for i in self.idx_stds]) # untransformed

        # grad KL of first loc parameter
        1/self.model.layers[0].w_scale_prior**2 * means_param[0]
        means[0]

        # grad KL of first scale parameter
        #stds_param_0_trans = self.model.layers[0].transform(stds_param[0])
        logistic = lambda z: 1/(1+torch.exp(-z))
        softplus = lambda z: torch.log(1+torch.exp(z))
        sigma = stds_param[0]
    
        1/self.model.layers[0].w_scale_prior**2*softplus(sigma)*logistic(sigma) - logistic(sigma)/softplus(sigma)
        stds[0] #gradient of untransformed parameter

        ###
        '''

        self.optimizer.zero_grad()

        return means, stds

class FullWeightsRecorder(Callback):
    def __init__(self):
        super(FullWeightsRecorder).__init__()

        self.transform = nn.Softplus()
    
    def on_train_begin(self, n_epochs, metrics):
        
        # parameter indices
        pnames = [name for name, param in self.model.named_parameters()]
        self.idx_means = [i for (i, pname) in enumerate(pnames) if 'loc' in pname]
        self.idx_stds = [i for (i, pname) in enumerate(pnames) if 'scale' in pname]
        
        # initial parameters
        means, stds = self.flatten_params()

        # allocate space
        self.means = [means]
        self.stds = [stds] # transformed

    def on_epoch_end(self, epoch, history):
        means, stds = self.flatten_params()

        self.means.append(means)
        self.stds.append(stds)

    def on_train_end(self, epoch, history):
        self.means = torch.cat([m.reshape(1,-1) for m in self.means],0)
        self.stds = torch.cat([m.reshape(1,-1) for m in self.stds],0)
        
    def flatten_params(self):
        parameters = [p.detach().clone().reshape(-1) for p in self.model.parameters()]    
        means = torch.cat([parameters[i] for i in self.idx_means])        
        stds = self.transform(torch.cat([parameters[i] for i in self.idx_stds])) 
        
        return means, stds


