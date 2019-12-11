#!/usr/bin/env python3
import argparse
import datetime
import errno
import os
import sys

import torch
import torch.nn.functional as F
from livelossplot import PlotLosses
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.charCNN.auxiliary import count_trainable_parameters, make_MJFF_data_loader, save_checkpoint
from src.charCNN.metric import print_f_score
from src.charCNN.model import CharCNN


def train(train_loader, dev_loader, model, args):

    # optimization scheme
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr=args.lr)

    # continue training from checkpoint model
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = torch.load(args.continue_from)
        start_epoch = checkpoint["epoch"]
        start_iter = checkpoint.get("iter", None)
        best_acc = checkpoint.get("best_acc", None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 1
        else:
            start_iter += 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 1
        start_iter = 1
        best_acc = None

    # dynamic learning scheme
    if args.dynamic_lr and args.optimizer != "Adam":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.decay_factor, last_epoch=-1
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=1e-3)

    # gpu
    if args.cuda:
        assert torch.cuda.is_available()
        # model = torch.nn.DataParallel(model).cuda()  # Only use if parallel GPU support
        model.cuda()

    # Call live-loss plots
    liveloss = PlotLosses()

    # Call train on pytorch model
    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        logs = {}  # For live-loss plotting
        running_loss = 0.0
        running_corrects = 0
        if args.dynamic_lr and args.optimizer != "Adam":
            scheduler.step()
        for i_batch, data in enumerate(train_loader, start=start_iter):
            # X, y
            inputs, target = data
            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            inputs = Variable(inputs)
            target = Variable(target)
            # Get sentence classification from model
            estimated_label = model(inputs)
            # Calculate the loss
            loss = F.nll_loss(estimated_label, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

            # For live plotting and learning checks
            corrects = (torch.max(estimated_label, 1)[1].view(target.size()).data == target.data).sum()
            # Get running score
            running_corrects += corrects
            # Get running loss
            running_loss += loss.detach() * inputs.size(0)
            # Get accuracy as function of corrects
            accuracy = 100.0 * corrects / args.batch_size

            if args.verbose:
                print("\nTargets, Predicates")
                print(
                    torch.cat(
                        (
                            target.unsqueeze(1),
                            torch.unsqueeze(torch.max(estimated_label, 1)[1].view(target.size()).data, 1),
                        ),
                        1,
                    )
                )
                print("\nLogit")
                print(estimated_label)

            if i_batch % args.log_interval == 0 and args.plot_live is False:
                print(
                    "Epoch[{}] Batch[{}] - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{})".format(
                        epoch,
                        i_batch,
                        loss.data,
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        accuracy,
                        corrects,
                        args.batch_size,
                    )
                )
            if i_batch % args.val_interval == 0:
                val_loss, val_acc, _, _ = eval(dev_loader, model, epoch, i_batch, optimizer, args)
            i_batch += 1

        # Get per-epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = running_corrects.float() / len(train_loader.dataset)

        # For live plotting
        logs["log loss"] = epoch_loss.item()
        logs["accuracy"] = epoch_accuracy.item()

        if args.checkpoint and epoch % args.save_interval == 0:
            file_path = "%s/CharCNN_epoch_%d.pth.tar" % (args.save_folder, epoch)
            print("\n\r=> saving checkpoint model to %s" % file_path)
            save_checkpoint(
                model, {"epoch": epoch, "optimizer": optimizer.state_dict(), "best_acc": best_acc}, file_path
            )

        # Validation
        val_loss, val_acc, val_running_loss, val_running_acc = eval(dev_loader, model, epoch, i_batch, optimizer, args)
        # For live plotting
        logs["val_log loss"] = val_running_loss.item()
        logs["val_accuracy"] = val_running_acc.item()

        # Save best validation epoch model
        if best_acc is None or val_acc > best_acc:
            file_path = "%s/CharCNN_best.pth.tar" % (args.save_folder)
            print("\r=> found better validated model, saving to %s" % file_path)
            save_checkpoint(
                model, {"epoch": epoch, "optimizer": optimizer.state_dict(), "best_acc": best_acc}, file_path
            )
            best_acc = val_acc
        print("\n")

        if args.plot_live:
            # Plot epoch metrics live
            liveloss.update(logs)
            liveloss.draw()


def eval(data_loader, model, epoch_train, batch_train, optimizer, args):

    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    for i_batch, data in enumerate(data_loader):
        inputs, target = data
        running_loss = 0.0
        size += len(target)
        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        with torch.no_grad():
            inputs = Variable(inputs)
            target = Variable(target)
            estimated_label = model(inputs)
            predicates = torch.max(estimated_label, 1)[1].view(target.size()).data
            # Loss
            loss = F.nll_loss(estimated_label, target)
            # loss = F.nll_loss(estimated_label, target, size_average=False).data
            # Running loss updated
            running_loss += loss.detach() * inputs.size(0)
            accumulated_loss += loss
            # Correctly labelled sentences, update total
            corrects += (torch.max(estimated_label, 1)[1].view(target.size()).data == target.data).sum()
            predicates_all += predicates.cpu().numpy().tolist()
            target_all += target.data.cpu().numpy().tolist()
            if args.cuda:
                torch.cuda.synchronize()

    # For live plotting
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = corrects.float() / len(data_loader.dataset)

    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    print(
        "\nEvaluation - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{}) ".format(
            avg_loss, optimizer.state_dict()["param_groups"][0]["lr"], accuracy, corrects, size
        )
    )
    print_f_score(predicates_all, target_all)
    print("\n")
    if args.log_result:
        with open(os.path.join(args.save_folder, "result.csv"), "a") as r:
            r.write(
                "\n{:d},{:d},{:.5f},{:.2f},{:f}".format(
                    epoch_train, batch_train, avg_loss, accuracy, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )

    return avg_loss, accuracy, epoch_loss, epoch_accuracy


def run(args):
    """
    General method which runs a full analysis using the charCNN model on the MJFF data.

    Parameters
    ----------
    args : "class"
        Class-like dict which holds all the config paramters.
    """

    # Load train data
    # XXX: there _may_ be a problem since the label is an int right now and not a str
    train_dataset, train_loader = make_MJFF_data_loader(
        args.train_path,
        args.alphabet_path,
        args.max_sample_length,
        args.load_very_long_sentences,
        args.batch_size,
        args.num_workers,
    )

    # Load dev data
    dev_dataset, dev_loader = make_MJFF_data_loader(
        args.val_path,
        args.alphabet_path,
        args.max_sample_length,
        args.load_very_long_sentences,
        args.batch_size,
        args.num_workers,
    )

    # Feature length (this is just the same as Zhang's original alphabet)
    args.set("num_features", len(train_dataset.alphabet))

    # Get class weights
    class_weight, num_class_train = train_dataset.get_class_weight()
    _, num_class_dev = dev_dataset.get_class_weight()

    # When you have an unbalanced training set
    if args.class_weight != None:
        args.set("class_weight", torch.FloatTensor(class_weight).sqrt_())
        if args.cuda:
            args.set("class_weight", args.class_weight.cuda())

    print("\nNumber of training samples: {}".format(str(train_dataset.__len__())))
    for i, c in enumerate(num_class_train):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))
    print("\nNumber of developing samples: {}".format(str(dev_dataset.__len__())))
    for i, c in enumerate(num_class_dev):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))

    # Create a folder to save the model within
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print("\n Directory for saving results, already exists.")
        else:
            raise

    # Configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace("_", " ")).ljust(25) + "{}".format(value))

    # Log result
    if args.log_result:
        with open(os.path.join(args.save_folder, "result.csv"), "w") as r:
            r.write("{:s},{:s},{:s},{:s},{:s}".format("epoch", "batch", "loss", "acc", "lr"))

    # Load the model
    model = CharCNN(args)
    print(model)
    print("\nNumber of trainable model parameters: %d\n" % count_trainable_parameters(model))

    # Run the actual training
    train(train_loader, dev_loader, model, args)

