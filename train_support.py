import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import neptune.new as neptune
from matplotlib import pyplot as plt

from datetime import datetime
import sys
import os
import glob

from utils import *

CONTENT_LAYERS = ['block5_conv2']
OPTIMIZER_DICT = {
                    "Adadelta" : tf.keras.optimizers.Adadelta, 
                    "Adagrad" : tf.keras.optimizers.Adagrad, 
                    "Adam" : tf.keras.optimizers.Adam, 
                    "Adamax" : tf.keras.optimizers.Adamax, 
                    "Ftrl" : tf.keras.optimizers.Ftrl, 
                    "Nadam" : tf.keras.optimizers.Nadam, 
                    "SGD" : tf.keras.optimizers.SGD 
}
OPTIMIZER_ARG = {
                    "Adadelta" : {'rho': 0.95, 'epsilon' : 1e-07, 'name' : 'Adadelta'}, 
                    "Adagrad" : tf.keras.optimizers.Adagrad, 
                    "Adam" : {'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-07, 'amsgrad' : False, 'name' : 'Adam'},
                    "Adamax" : tf.keras.optimizers.Adamax, 
                    "Ftrl" : tf.keras.optimizers.Ftrl, 
                    "Nadam" : tf.keras.optimizers.Nadam, 
                    "SGD" : tf.keras.optimizers.SGD 
}

def get_optimizer_arg(optimizer_name):
    '''
    Returns all the optimizer arguments
    Arguments --
        optimizer_name -- string | one of ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "SGD"]
    '''
    return OPTIMIZER_ARG[optimizer_name]

                
def train_siamese(train_dataset, dev_dataset, lr, batch_size, epochs, model, train_step, test_step, compute_cost, optimizer_name, save_dir='checlpoints/save', 
                checkpoint_path='', start_epoch=0, eval_metrics = {}, optimizer_arg ={}, dataset_name = '', loss_name = '', model_name = '', run = None, show_results=-1, save_image=False):
    '''
    This function gets executed on executing the script.
    
    Args ::
        train_dataset -- Training set
        dev_dataset -- Dev set
        lr -- float or float tensor or learning_rate_function | learning rate
        batch_size -- int | number of batches used
        epochs -- int or int tensor | number of epochs
        model -- tf.keras.nn.Model
        optimizer_name -- string | one of ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "SGD"]
        train_step -- function | train for 1 iteration | takes 6 arguments 
                                        minibatch_X -- feature minibatch
                                        minibatch_Y -- label minibatch
                                        model -- CV model
                                        compute_cost -- cost function
                                        optimizer -- optimizer
                                        vgg -- vgg model if use_vgg is true
        test_step -- function | test for 1 iteration | takes 5 arguments 
                                        minibatch_X -- feature minibatch
                                        minibatch_Y -- label minibatch
                                        model -- CV model
                                        compute_cost -- cost function
                                        vgg -- vgg model if use_vgg is true
        compute_cost - function | calculates loss | takes 3 arguments
                                        Y - gt labels
                                        Y_pred - predicted labels
                                        vgg -- vgg model if use_vgg is true
        save_dir -- str | default 'checkpoint/save' | dir where checkpoints to be saved
        checkpoint_path - str | default '' | specifies the saved checkpoint path to restore
                                        the model from
        start_epoch -- int | default 0 | starting epoch
        eval_metrics -- python dict | default {} | dict of metrics to be evaluated |
                                            the dict should contain eval step function as values and 
                                            metric name as keys
        optimizer_arg -- python dict | default {} | the dict should contain parameters to the optimizer function 
                                            other than lr as  key value  eval step function as values and 
                                            metric name as keys
        dataset_name -- str | default '' | name of the dataset used in trainiing
        loss_name -- str | default '' | name of the loss(es) used in trainiing
        model_name -- str | default '' | name of the model used in trainiing
        run -- neptune logger | can be got using neptune.init()
        show_results -- int | default -1 | if set between 0 to epochs; computes
                                    metrics and displayes results from dev set
                                    in that intervals
    '''
    ## allow gpu growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    gpus = ['gpu:'+gpu.name[-1] for gpu in gpus]
    print(f'GPUs : {gpus}')

    ## hyperparameters
    PARAMS = {
            'start-lr' : lr,
            'batch-size' : batch_size,
            'dataset-name' : dataset_name,
            'loss-name' : loss_name,
            'model-name' : model_name
    }

    
    ## initial neptune log
    if run is not None:
        run["hyper-parameters/general"] = PARAMS


    checkpoint_dir = save_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ## initialize data loader
    n_minibatches = train_dataset.count_minibatches()
    n_minibatches_dev = dev_dataset.count_minibatches()

    print(f'Total number of training examples = {train_dataset.m}')

    ## creating optimizer
    optimizer_arg_default = OPTIMIZER_ARG[optimizer_name] if optimizer_name in OPTIMIZER_ARG.keys() else {}

    for k, v in optimizer_arg.items():
        if k in optimizer_arg_default.keys():
            optimizer_arg_default[k] = v
    
    optimizer_arg = optimizer_arg_default
    
    if optimizer_name not in OPTIMIZER_DICT.keys():
        print(f'Invalid optimizer option')
        print(f'Optimizer should be one of  : {OPTIMIZER_DICT.keys()}')

    optimizer = create_optimizer(OPTIMIZER_DICT[optimizer_name], lr, optimizer_arg)

    ## log optimizer hyperpaprameters
    if run is not None:
        run["hyper-parameters/optimizer"] =  optimizer_arg

    ## load checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)
    if checkpoint_path != '':
        if os.path.exists(checkpoint_path):
            print(f'Restoring checkpoint from {checkpoint_path}', end='\r')
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
            # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

            run['logs/checkpoint_path/restore_path'] = checkpoint_path
            # for name in os.listdir(checkpoint_path):
            #    run['logs/checkpoint_path/name'].upload(os.path.join(checkpoint_path, name) 

    print(f'Start epoch - {start_epoch} | End epoch - {start_epoch + epochs}')
    print(f'Number of minibatches in training set - {n_minibatches}')
    print('Starting training...')
    
    for epoch in range(start_epoch, start_epoch+epochs):
        costs = []
        dev_costs = []
        dev_metric = {k : [] for k, v in eval_metrics.items()}
        minibatch_cost = 0
        devminibatch_cost = 0
        devminibatch_metric = {k : 0 for k, v in eval_metrics.items()}

        ## iterate over minibatches
        for iteration in range(n_minibatches):
            ## fetch one minibatch
            data_dict = train_dataset.get_data()
            minibatch_X = data_dict['images']
            minibatch_Y = data_dict['labels']


            ## train the model for one iteration
            temp_cost, Y_pred = train_step(minibatch_X, minibatch_Y, model, compute_cost, optimizer)            
            minibatch_cost += temp_cost

            step = (iteration + 1) + (epoch * n_minibatches)
            step_lr = lr(step) if not isinstance(lr, float) else lr

            ## save a sample image to see progress
            if save_image and step % 100 == 0:
                if isinstance(minibatch_X, list):
                    i = 0
                    for im_batch in minibatch_X:
                        plt.imsave(f'save/{step}_{i}.png', tf.clip_by_value(im_batch[0], 0, 1).numpy()[0])
                        i += 1
                else:
                    plt.imsave(f'save/{step}.png', tf.clip_by_value(minibatch_X[0], 0, 1).numpy()[0])

                with open(f'save/{step}_{Y_pred[0]}', 'w') as f:
                    f.writelines(f'{Y_pred[0]}')
               

            if iteration > 0:
                sys.stdout.write("\033[K")
            print(f'{iteration + 1}/{n_minibatches} minibatches processed | {step} iterations | cost - {temp_cost} | lr - {step_lr}', end='\r')

        ## track cost
        costs.append(minibatch_cost) # /len(minibatch_cost))
        minibatch_cost = 0
        sys.stdout.write("\033[K")
        print(f'Training set cost after {epoch} epochs =  {costs[-1]}')

        if run is not None:
            run["train/loss"].log(costs[-1])
            run["train/epoch"].log(epoch)


        ## save the checkpoint
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint.save(os.path.join(checkpoint_dir, str(epoch)))

        ## evaluate if show_result in greater than 0 and after every show_result epochs
        if show_results > 0 and epoch % show_results == 0:
            ## iterate over dev set
            for iteration in range(n_minibatches_dev):
                ## fetch one minibatch
                data_dict = dev_dataset.get_data()
                minibatch_X = data_dict['images']
                minibatch_Y = data_dict['labels']

                ## calculate cost and Y_pred
                temp_cost, Y_pred = test_step(minibatch_X, minibatch_Y, model, compute_cost)
                devminibatch_cost += temp_cost

                ## calculate metrics
                for kmetric, vmetric in eval_metrics.items():
                    devminibatch_metric[kmetric] += vmetric(Y_pred, minibatch_Y)

                if iteration > 0:
                    sys.stdout.write("\033[K")
                print(f'{iteration + 1}/{n_minibatches} minibatches processed | {step} iterations | dev cost - {temp_cost}', end='\r')
            
            ## track cost and PSNR
            dev_costs.append(devminibatch_cost) # /len(minibatch_cost))
            for kmetric, vmetric in eval_metrics.items():
                dev_metric[kmetric].append(devminibatch_metric[kmetric])
            devminibatch_cost = 0
            devminibatch_metric =  {k : 0 for k, v in eval_metrics.items()}
            sys.stdout.write("\033[K")
            print(f'Dev set cost after {epoch} epochs =  {dev_costs[-1]}') #'| PSNR = {dev_psnr[-1]}')

            if run is not None:
                run['dev/loss'].log(dev_costs[-1])
                for kmetric, vmetric in eval_metrics.items():
                    run[f'dev/{kmetric}'].log(dev_metric[kmetric][-1])



def train_siamese_one_shot(train_dataset, dev_dataset, lr, batch_size, epochs, model, train_step, test_step, compute_cost, optimizer_name, save_dir='checlpoints/save', 
                checkpoint_path='', start_epoch=0, eval_metrics = {}, optimizer_arg ={}, dataset_name = '', loss_name = '', model_name = '', run = None, show_results=-1, save_image=False):
    '''
    This function gets executed on executing the script.
    
    Args ::
        train_dataset -- Training set
        dev_dataset -- Dev set
        lr -- float or float tensor or learning_rate_function | learning rate
        batch_size -- int | number of batches used
        epochs -- int or int tensor | number of epochs
        model -- tf.keras.nn.Model
        optimizer_name -- string | one of ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "SGD"]
        train_step -- function | train for 1 iteration | takes 6 arguments 
                                        minibatch_X -- feature minibatch
                                        minibatch_Y -- label minibatch
                                        model -- CV model
                                        compute_cost -- cost function
                                        optimizer -- optimizer
                                        vgg -- vgg model if use_vgg is true
        test_step -- function | test for 1 iteration | takes 5 arguments 
                                        minibatch_X -- feature minibatch
                                        minibatch_Y -- label minibatch
                                        model -- CV model
                                        compute_cost -- cost function
                                        vgg -- vgg model if use_vgg is true
        compute_cost - function | calculates loss | takes 3 arguments
                                        Y - gt labels
                                        Y_pred - predicted labels
                                        vgg -- vgg model if use_vgg is true
        save_dir -- str | default 'checkpoint/save' | dir where checkpoints to be saved
        checkpoint_path - str | default '' | specifies the saved checkpoint path to restore
                                        the model from
        start_epoch -- int | default 0 | starting epoch
        eval_metrics -- python dict | default {} | dict of metrics to be evaluated |
                                            the dict should contain eval step function as values and 
                                            metric name as keys
        optimizer_arg -- python dict | default {} | the dict should contain parameters to the optimizer function 
                                            other than lr as  key value  eval step function as values and 
                                            metric name as keys
        dataset_name -- str | default '' | name of the dataset used in trainiing
        loss_name -- str | default '' | name of the loss(es) used in trainiing
        model_name -- str | default '' | name of the model used in trainiing
        run -- neptune logger | can be got using neptune.init()
        show_results -- int | default -1 | if set between 0 to epochs; computes
                                    metrics and displayes results from dev set
                                    in that intervals
    '''
    ## allow gpu growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    gpus = ['gpu:'+gpu.name[-1] for gpu in gpus]
    print(f'GPUs : {gpus}')

    ## hyperparameters
    PARAMS = {
            'start-lr' : lr,
            'batch-size' : batch_size,
            'dataset-name' : dataset_name,
            'loss-name' : loss_name,
            'model-name' : model_name
    }

    
    ## initial neptune log
    if run is not None:
        run["hyper-parameters/general"] = PARAMS


    checkpoint_dir = save_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ## initialize data loader
    n_minibatches = train_dataset.count_minibatches()
    n_minibatches_dev = dev_dataset.count_minibatches()

    print(f'Total number of training examples = {train_dataset.m}')

    ## creating optimizer
    optimizer_arg_default = OPTIMIZER_ARG[optimizer_name] if optimizer_name in OPTIMIZER_ARG.keys() else {}

    for k, v in optimizer_arg.items():
        if k in optimizer_arg_default.keys():
            optimizer_arg_default[k] = v
    
    optimizer_arg = optimizer_arg_default
    
    if optimizer_name not in OPTIMIZER_DICT.keys():
        print(f'Invalid optimizer option')
        print(f'Optimizer should be one of  : {OPTIMIZER_DICT.keys()}')

    optimizer = create_optimizer(OPTIMIZER_DICT[optimizer_name], lr, optimizer_arg)

    ## log optimizer hyperpaprameters
    if run is not None:
        run["hyper-parameters/optimizer"] =  optimizer_arg

    ## load checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)
    if checkpoint_path != '':
        if os.path.exists(checkpoint_path):
            print(f'Restoring checkpoint from {checkpoint_path}', end='\r')
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
            # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

            run['logs/checkpoint_path/restore_path'] = checkpoint_path
            # for name in os.listdir(checkpoint_path):
            #    run['logs/checkpoint_path/name'].upload(os.path.join(checkpoint_path, name) 

    print(f'Start epoch - {start_epoch} | End epoch - {start_epoch + epochs}')
    print(f'Number of minibatches in training set - {n_minibatches}')
    print('Starting training...')
    
    for epoch in range(start_epoch, start_epoch+epochs):
        costs = []
        dev_costs = []
        dev_metric = {k : [] for k, v in eval_metrics.items()}
        minibatch_cost = 0
        devminibatch_cost = 0
        devminibatch_metric = {k : 0 for k, v in eval_metrics.items()}

        ## iterate over minibatches
        for iteration in range(n_minibatches):
            step = (iteration + 1) + (epoch * n_minibatches)

            ## fetch one minibatch
            data_dict = train_dataset.get_data()
            minibatch_X = data_dict['images']
            minibatch_Y = data_dict['labels']
            anchor, pos, neg = minibatch_X

            pos_pair = [anchor, pos]
            neg_pair = [anchor, neg]
            pos_y = np.ones((batch_size, 1))
            neg_y = np.zeros((batch_size, 1))

            ## train the model for one iteration
            temp_cost, Y_pred = train_step(pos_pair, pos_y, model, compute_cost, optimizer)         
            minibatch_cost += temp_cost / 2

            if iteration > 0:
                sys.stdout.write("\033[K")
            print(f'{iteration + 1}/{n_minibatches} minibatches processed | {step} iterations | cost - {temp_cost}', end='\r')

            temp_cost, Y_pred = train_step(neg_pair, neg_y, model, compute_cost, optimizer)
            minibatch_cost += temp_cost / 2

            if iteration > 0:
                sys.stdout.write("\033[K")
            print(f'{iteration + 1}/{n_minibatches} minibatches processed | {step} iterations | cost - {temp_cost}', end='\r')
            
            step_lr = lr(step) if not isinstance(lr, float) else lr

            ## save a sample image to see progress
            if save_image and step % 100 == 0:
                if isinstance(minibatch_X, list):
                    i = 0
                    for im_batch in minibatch_X:
                        plt.imsave(f'save/{step}_{i}.png', tf.clip_by_value(im_batch[0], 0, 1).numpy()[0])
                        i += 1
                else:
                    plt.imsave(f'save/{step}.png', tf.clip_by_value(minibatch_X[0], 0, 1).numpy()[0])

                with open(f'save/{step}_{Y_pred[0]}', 'w') as f:
                    f.writelines(f'{Y_pred[0]}')
               
            

        ## track cost
        costs.append(minibatch_cost) # /len(minibatch_cost))
        minibatch_cost = 0
        sys.stdout.write("\033[K")
        print(f'Training set cost after {epoch} epochs =  {costs[-1]}')

        if run is not None:
            run["train/loss"].log(costs[-1])
            run["train/epoch"].log(epoch)


        ## save the checkpoint
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint.save(os.path.join(checkpoint_dir, str(epoch)))

        ## evaluate if show_result in greater than 0 and after every show_result epochs
        if show_results > 0 and epoch % show_results == 0:
            ## iterate over dev set
            for iteration in range(n_minibatches_dev):
                ## fetch one minibatch
                data_dict = dev_dataset.get_data()
                minibatch_X = data_dict['images']
                minibatch_Y = data_dict['labels']

                ## calculate cost and Y_pred
                temp_cost, Y_pred = test_step(minibatch_X, minibatch_Y, model, compute_cost)
                devminibatch_cost += temp_cost

                ## calculate metrics
                for kmetric, vmetric in eval_metrics.items():
                    devminibatch_metric[kmetric] += vmetric(Y_pred, minibatch_Y)

                if iteration > 0:
                    sys.stdout.write("\033[K")
                print(f'{iteration + 1}/{n_minibatches} minibatches processed | {step} iterations | dev cost - {temp_cost}', end='\r')
            
            ## track cost and PSNR
            dev_costs.append(devminibatch_cost) # /len(minibatch_cost))
            for kmetric, vmetric in eval_metrics.items():
                dev_metric[kmetric].append(devminibatch_metric[kmetric])
            devminibatch_cost = 0
            devminibatch_metric =  {k : 0 for k, v in eval_metrics.items()}
            sys.stdout.write("\033[K")
            print(f'Dev set cost after {epoch} epochs =  {dev_costs[-1]}') #'| PSNR = {dev_psnr[-1]}')

            if run is not None:
                run['dev/loss'].log(dev_costs[-1])
                for kmetric, vmetric in eval_metrics.items():
                    run[f'dev/{kmetric}'].log(dev_metric[kmetric][-1])

