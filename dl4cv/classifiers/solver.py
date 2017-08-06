from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    weights = torch.cuda.FloatTensor([0.0122442771, 0.0127766370, 0.0050193931, 0.0316373869, 0.0270743022, 0.0228914746, 0.0164271047, 0.0080614495, 0.0244885543,
                                0.0165031561, 0.0204578295, 0.0222070119, 0.0244885543, 0.0185565442, 0.0083656552, 0.1076127462, 0.0444900753, 0.0046391361,
                                0.0174157731, 0.0259335311, 0.0190889041, 0.01832839, 0.0031181078, 0.0112556088, 0.0870028139, 0.0330823637, 0.0186325956,
                                0.0096585292, 0.0367328314, 0.018708647, 0.0341470834, 0.0184804928, 0.0342991862, 0.0334626207, 0.0247927599, 0.0922503612,
                                0.0064643699, 0.0292037417])

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func(weight=self.weights)
        self.loss_func_val = loss_func()

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []

        self.train_loss_history_per_epoch = []
        self.val_loss_history_per_epoch = []

        self.train_acc_history = []
        self.val_acc_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, print_summary=False):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        # filter(lambda p: p.requires_grad, model.parameters())
        use_gpu = torch.cuda.is_available()

        print 'START TRAIN.'
        ############################################################################
        # TODO:                                                                    #
        # Write your own personal training method for our solver. In Each epoch    #
        # iter_per_epoch shuffled training batches are processed. The loss for     #
        # each batch is stored in self.train_loss_history. Every log_nth iteration #
        # the loss is logged. After one epoch the training accuracy of the last    #
        # mini batch is logged and stored in self.train_acc_history.               #
        # We validate at the end of each epoch, log the result and store the       #
        # accuracy of the entire validation set in self.val_acc_history.           #
        #
        # Your logging should like something like:                                 #
        #   ...                                                                    #
        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #
        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #
        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
        #   ...                                                                    #
        ############################################################################
        for epoch in range(num_epochs):
            total_train  = 0
            correct_train = 0
            train_loss_per_epoch = 0
            
            #Training Loop
            
            for i, data in enumerate(train_loader, 0):
                # get the inputs, wrap them in Variable
                input, label = data
                if use_gpu:
                    input = input.cuda()
                    label = label.cuda()
                    model = model.cuda()
                
                inputs, labels = Variable(input), Variable(label)
                
                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optim.step()
                
                _,predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == label).sum()

                # Storing train loss history per iteration
                self.train_loss_history.append(loss.data[0])

                train_loss_per_epoch += loss.data[0]

                if (i+1) % log_nth == 0:
                    if print_summary:
                        print ('[Iteration %d/%d] Train loss: %0.4f') % \
                              (i, iter_per_epoch, self.train_loss_history[-1])

                #if (i+1) % iter_per_epoch == 0:
                #    #storing train accuracy history per epoch
                #    self.train_acc_history.append(correct_train/float(total_train))

                #    #storing train loss history per epoch
                #    self.train_loss_history_per_epoch.append(train_loss_per_epoch)

                #    print ('[Epoch %d/%d] Train acc/loss: %0.4f/%0.4f') % \
                #          (epoch, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1])

            #storing train accuracy history per epoch
            self.train_acc_history.append(correct_train/float(total_train))

            #storing train loss history per epoch
            self.train_loss_history_per_epoch.append(train_loss_per_epoch)

            if print_summary:
                print ('[Epoch %d/%d] Train acc/loss: %0.4f/%0.4f') % \
                      (epoch, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1])

            #Validation Loop
            
            correct_val = 0
            val_size = 0
            loss_val = 0
            val_loss_per_epoch = 0

            for i, data_val in enumerate(val_loader, 0):
                # get the inputs, wrap them in Variable
                input_val, label_val = data_val
                
                if use_gpu:
                    input_val = input_val.cuda()
                    label_val = label_val.cuda()
                
                inputs_val, labels_val = Variable(input_val), Variable(label_val)
                output_val = model(inputs_val)
                loss_val = self.loss_func_val(output_val, labels_val)
                _,predicted_val = torch.max(output_val.data, 1)
                val_size += label_val.size(0)
                correct_val += (predicted_val == label_val).sum()

                # storing val loss history per iteration
                self.val_loss_history.append(loss_val.data[0])

                val_loss_per_epoch += loss_val.data[0]

            #storing val accuracy history per epoch
            self.val_acc_history.append(correct_val/float(val_size))

            #storing val loss history per epoch
            self.val_loss_history_per_epoch.append(val_loss_per_epoch)

            if print_summary:
                print ('[Epoch %d/%d] Val acc/loss: %0.4f/%0.4f') % \
                      (epoch, num_epochs, self.val_acc_history[-1], loss_val.data[0])

        print ('Training acc/loss: %0.4f/%0.4f && Validation acc/loss: %0.4f/%0.4f') % \
              (self.train_acc_history[-1], self.train_loss_history_per_epoch[-1], self.val_acc_history[-1], self.val_loss_history_per_epoch[-1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'
