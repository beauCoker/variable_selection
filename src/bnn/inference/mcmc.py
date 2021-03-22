# standard library imports

# package imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_probability as tfp

# local imports

def hmc_tf(unnormalized_log_prob, init_values, num_results = int(10e3), num_burnin_steps = int(1e3), num_leapfrog_steps=3, step_size=1.):
    '''
    Inputs:
        unnormalized_log_prob: python callable (tensor inputs and outputs)
        init_values: initial parameter values (numpy array)
        [various optional hmc arguments]
    Outputs:
        samples: posterior samples (numpy array)
        is_accepted: acceptance rate (scalar)
    '''

    # Run HMC
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=3,
            step_size=1.),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    if isinstance(init_values, list):
        init_values_tf = [tf.convert_to_tensor(v) for v in init_values]
    else:
        init_values_tf = tf.convert_to_tensor(init_values)

    # Run the chain (with burn-in).
    @tf.function
    def run_chain():
        # Run the chain (with burn-in).
        samples, is_accepted = tfp.mcmc.sample_chain(
              num_results=num_results,
              num_burnin_steps=num_burnin_steps,
              current_state=init_values_tf,
              kernel=adaptive_hmc,
              seed=1,
              trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
        
        # simple summary stats
        if isinstance(samples, list):
            sample_mean = [tf.reduce_mean(s, 0) for s in samples]
            sample_stddev = [tf.math.reduce_std(s) for s in samples]
        else:
            sample_mean = tf.reduce_mean(samples, 0)
            sample_stddev = tf.math.reduce_std(samples)

        is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
        
        return sample_mean, sample_stddev, is_accepted, samples

    sample_mean, sample_stddev, is_accepted, samples = run_chain()

    # convert to numpy for output
    if isinstance(samples, list):
        samples_np = [np.squeeze(s.numpy()) for s in samples]
    else:
        samples_np = np.squeeze(samples.numpy())

    is_accepted_np = np.squeeze(is_accepted.numpy())

    return samples_np, is_accepted_np


class ModelTrainer(object):
    '''
    Wrapper for models

    model should have the following methods:
        loss(x, y)
        init_parameters(seed)
    '''
    def __init__(self, model):
        super(ModelTrainer).__init__()
        self.model = model

    def train_random_restarts(self, n_restarts, n_epochs, x, y, optimizer, scheduler=None, n_rep_opt=1, callback_list=None, **kwargs_loss):
        seeds = torch.zeros(n_restarts).long().random_(0, 100*n_restarts)
        loss_best = float('inf')
        for i, seed in enumerate(seeds):
            print('random restart [%d/%d]' % (i, n_restarts))
            self.model.init_parameters(seed)
            
            history = self.train(n_epochs, x, y, optimizer, scheduler, n_rep_opt, callback_list, **kwargs_loss)
            
            if history['loss'][-1] < loss_best:
                # what old model was reloaded? Then history['loss'][-1] isn't the loss of the final model
                i_best = i
                seed_best = seed
                state_dict_best = self.model.state_dict().copy()
                history_best = history.copy()

        print('best was restart %d (seed=%d)' % (i_best, seed_best))
        self.model.load_state_dict(state_dict_best)
        return history_best

    def train(self, n_epochs, x, y, optimizer, scheduler=None, n_rep_opt=1, callback_list=None, **kwargs_loss):
        '''
        '''
        with torch.no_grad():

            # initialize history
            _, metrics_initial = self.model.loss(x, y, **kwargs_loss)
            history = {}
            for key, value in metrics_initial.items():
                history[key] = [value]

            # set up callbacks
            if callback_list is not None:
                callback_list = callbacks.CallbackList(callback_list, self.model, optimizer)
                callback_list.on_train_begin(n_epochs, metrics_initial)

        print('training...')
        for epoch in range(1, n_epochs+1):

            for i in range(n_rep_opt):
                optimizer.zero_grad()

                loss, metrics = self.model.loss(x, y, **kwargs_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
                optimizer.step()

            # checking model (saving, early stopping, etc.)
            with torch.no_grad():

                if scheduler is not None:
                    scheduler.step(epoch)

                # update history
                [history[key].append(value) for key, value in metrics.items()]

                if callback_list is not None:
                    flags = callback_list.on_epoch_end(epoch, history)
                    if any(flags):
                        break

        with torch.no_grad():
            if callback_list is not None:
                callback_list.on_train_end(epoch, history)

        return history





def train(model, optimizer, x, y, n_epochs, x_linear=None, n_warmup = 0, n_rep_opt=10, print_freq=None, frac_start_save=1, frac_lookback=0.5, path_checkpoint='./'):
    '''
    frac_lookback will only result in reloading early stopped model if frac_lookback < 1 - frac_start_save
    '''

    loss = torch.zeros(n_epochs)
    loss_best = torch.tensor(float('inf'))
    loss_best_saved = torch.tensor(float('inf'))
    saved_model = False
    model.precompute(x, x_linear)

    for epoch in range(n_epochs):

        # TEMPERATURE HARDECODED, NEED TO FIX
        #temperature_kl = 0. if epoch < n_epochs/2 else 1.0
        #temperature_kl = epoch / (n_epochs/2) if epoch < n_epochs/2 else 1.0
        temperature_kl = epoch / (n_epochs/10) if epoch < n_epochs/10 else 1.0
        #temperature_kl = 0. # SET TO ZERO TO IGNORE KL

        for i in range(n_rep_opt):

            l = model.loss(x, y, x_linear=x_linear, temperature=temperature_kl)

            # backward
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            optimizer.step()

            ##
            #print('------------- %d -------------' % epoch)
            #print('s     :', model.layer_in.s_loc.data)
            #print('grad  :', model.layer_in.s_loc.grad)

            #model.layer_in.s_loc.grad.zero_()
            #kl = model.layer_in.kl_divergence()
            #kl.backward()
            #if epoch > 500:
            #    breakpoint()
            #print('grad kl:', model.layer_in.s_loc.grad)
            ##

        loss[epoch] = l.item()

        with torch.no_grad():
            model.fixed_point_updates(x, y, x_linear=x_linear, temperature=1)

        # print state
        if print_freq is not None:
            if (epoch + 1) % print_freq == 0:
                model.print_state(x, y, epoch+1, n_epochs)

        # see if improvement made (only used if KL isn't tempered)
        if loss[epoch] < loss_best and temperature_kl==1.0:
            loss_best = loss[epoch]

        # save model
        if epoch > frac_start_save*n_epochs and loss[epoch] < loss_best_saved:
            print('saving mode at epoch = %d' % epoch)
            saved_model = True
            loss_best_saved = loss[epoch]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss[epoch],
            },  os.path.join(path_checkpoint, 'checkpoint.tar'))

        # end training if no improvement made in a while and more than half way done
        epoch_lookback = np.maximum(1, int(epoch - .25*n_epochs)) # lookback is 25% of samples by default
        if epoch_lookback > frac_start_save*n_epochs+1:
            loss_best_lookback = torch.min(loss[epoch_lookback:epoch+1])
            percent_improvement = (loss_best - loss_best_lookback)/torch.abs(loss_best) # positive is better
            if percent_improvement < 0.0:
                print('stopping early at epoch = %d' % epoch)
                break

    # reload best model if saving
    if saved_model:
        checkpoint = torch.load(os.path.join(path_checkpoint, 'checkpoint.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('reloading best model from epoch = %d' % checkpoint['epoch'])
        model.eval()


    return loss[:epoch]


def train_score(model, optimizer, x, y, n_epochs, x_linear=None, n_warmup = 0, n_rep_opt=10, print_freq=None, frac_start_save=1):
    loss = torch.zeros(n_epochs)
    loss_best = 1e9 # Need better way of initializing to make sure it's big enough
    model.precompute(x, x_linear)

    for epoch in range(n_epochs):

        # TEMPERATURE HARDECODED, NEED TO FIX
        #temperature_kl = 0. if epoch < n_epochs/2 else 1
        #temperature_kl = epoch / (n_epochs/2) if epoch < n_epochs/2 else 1
        temperature_kl = 0. # SET TO ZERO TO IGNORE KL

        for i in range(n_rep_opt):

            optimizer.zero_grad()

            model.compute_loss_gradients(x, y, x_linear=x_linear, temperature=temperature_kl)

            # backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()

        with torch.no_grad():
            model.fixed_point_updates(x, y, x_linear=x_linear, temperature=1)

        if epoch > frac_start_save*n_epochs and loss[epoch] < loss_best: 
            print('saving...')
            loss_best = loss[epoch]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss[epoch],
            }, 'checkpoint.tar')

        if print_freq is not None:
            if (epoch + 1) % print_freq == 0:
                model.print_state(x, y, epoch+1, n_epochs)

    return loss
