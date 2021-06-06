import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import ReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from file_organize import model_prediction, evaluate_regression_results, plot_and_save, scatter_plot
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

# Vanilla Neural Network
def model_vanilla_NN(x_train, y_train, save_image_path, BATCH_SIZE, EPOCHS, L_RATE, VAL_SPLIT = 0.1):
    # batch size: BATCH_SIZE
    # Number of training epochs: EPOCHS
    # Learning rate: L_RATE
    # Proportion of samples to hold out: VAL_SPLIT

    # Vanilla NN
    D = x_train.shape[1]

    # Model:
    model = Sequential([
        Dense(512, use_bias=False, input_shape=(D,)),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(128, use_bias=False),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(64, use_bias=False),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(32, use_bias=False),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(1)
    ])

    #print(model.summary())

    # Compile the model with MAE loss
    model.compile(tf.keras.optimizers.Adam(lr=L_RATE), loss='mean_absolute_error')

    # Fit the model
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=VAL_SPLIT,
                        verbose=0)

    # Plot the ELBO loss
    target_name = 'MAE on training set over source domain'
    #plot_and_save(history.history['loss'], target_name, save_image_path)

    target_name = 'MAE on test set over source domain'
    #plot_and_save(history.history['val_loss'], target_name, save_image_path)

    return model

# If only retrain last few layers, use this function.
def retrain_vanilla_NN(model, x_train, y_train, save_image_path, BATCH_SIZE, EPOCHS, L_RATE, VAL_SPLIT=0.1):

    for layer in model.layers[:-2]:
        layer.trainable = False

    for layer in model.layers:
        print(layer, layer.trainable)

    # Compile the model with MAE loss
    model.compile(tf.keras.optimizers.Adam(lr=L_RATE), loss='mean_absolute_error')

    # Fit the model
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=VAL_SPLIT,
                        verbose=0)

    target_name = 'MAE on training set over target domain'
    plot_and_save(history.history['loss'], target_name, save_image_path)

    target_name = 'MAE on test set over target domain'
    plot_and_save(history.history['val_loss'], target_name, save_image_path)

    return model


def train_DNN(current_years, num_years,
              x_train_source_domain, y_train_source_domain, x_train_target_domain, y_train_target_domain,
              x_test_source_domain, y_test_source_domain, x_test_target_domain, y_test_target_domain,
              save_filename, save_scatter_image_path, BATCH_SIZE, EPOCHS, L_RATE):

    print("Traning size from source domain:", x_train_source_domain.shape)
    print("Test size from source domain:", x_test_source_domain.shape)

    experiment_years = current_years[0:num_years]

    f = open(save_filename, "a+")
    f.write("year = %f pretrained = 0 " % (experiment_years[-1]))

    save_image_path = '_transfer_result/image/DNN_' + str(experiment_years[-1])
    model_before = model_vanilla_NN(x_train_source_domain, y_train_source_domain,
                                    save_image_path, BATCH_SIZE, EPOCHS, L_RATE)

    # evaluation
    y_train_pred_source_domain = model_prediction(model_before, x_train_source_domain)
    y_train_pred_target_domain = model_prediction(model_before, x_train_target_domain)
    y_test_pred_source_domain = model_prediction(model_before, x_test_source_domain)
    y_test_pred_target_domain = model_prediction(model_before, x_test_target_domain)

    print("Evaluation on source domain training set: ")
    RMSE_train_source_domain, R2_train_source_domain = evaluate_regression_results(y_train_source_domain,y_train_pred_source_domain)

    print("Evaluation on source domain training set: ")
    RMSE_train_target_domain, R2_train_target_domain = evaluate_regression_results(y_train_target_domain,y_train_pred_target_domain)

    print("Evaluation on source domain test set: ")
    RMSE_test_source_domain, R2_test_source_domain = evaluate_regression_results(y_test_source_domain,y_test_pred_source_domain)
    print("Evaluation on target domain test set: ")
    RMSE_test_target_domain, R2_test_target_domain = evaluate_regression_results(y_test_target_domain,y_test_pred_target_domain)
    print("__________________________________")

    f.write("%f %f %f %f %f %f %f %f" % (
    RMSE_train_source_domain, R2_train_source_domain, RMSE_train_target_domain, R2_train_target_domain,
    RMSE_test_source_domain, R2_test_source_domain,RMSE_test_target_domain, R2_test_target_domain))

    f.write("\n")
    f.close()

    # Save the model for refining
    save_model = 1
    if save_model == 1:
        save_model_path = '_transfer_result/model/DNN_' + str(experiment_years[-1]) + '.h5'
        model_before.save(save_model_path)

    save_scatter_plot = 1
    if save_scatter_plot == 1:
        # Target Domain
        reported = y_test_target_domain
        predicted = y_test_pred_target_domain
        RMSE = RMSE_test_target_domain
        R2 = R2_test_target_domain

        save_name = 'Target Domain DNN ' + str(experiment_years[-1])
        print("save DNN scatter plot.")
        scatter_plot(reported, predicted, RMSE, R2, save_name, save_scatter_image_path)
        # Source Domain
        reported = y_test_source_domain
        predicted = y_test_pred_source_domain
        RMSE = RMSE_test_source_domain
        R2 = R2_test_source_domain

        save_name = 'Target Domain DNN ' + str(experiment_years[-1])
        print("save DNN scatter plot.")
        scatter_plot(reported, predicted, RMSE, R2, save_name, save_scatter_image_path)



def semi_sup_retrain_DNN(current_years, num_years, kk,
                         x_train_source_domain, y_train_source_domain, x_train_target_domain, y_train_target_domain,
                         x_test_source_domain, y_test_source_domain, x_test_target_domain, y_test_target_domain,
                         save_filename, save_scatter_image_path, BATCH_SIZE, EPOCHS, L_RATE):
    experiment_years = current_years[0:num_years]

    # path to save image
    save_image_path = '_transfer_result/image/DNN_' + str(experiment_years[-1]) + '_' + str(int(kk))
    print(save_image_path)

    # Load the pretrained model
    save_model_path = '_transfer_result/model/DNN_' + str(experiment_years[-1]) + '.h5'
    model_before = keras.models.load_model(save_model_path)

    # organize semi-supervised data
    y_train_pred_target_domain_semi = model_prediction(model_before, x_train_target_domain)

    x_train_both = np.concatenate((x_train_source_domain, x_train_target_domain), axis=0)
    # Change 1
    # y_train_both = np.concatenate((y_train_source_domain, y_train_target_domain), axis=0)
    y_train_both = np.concatenate((y_train_source_domain, y_train_pred_target_domain_semi), axis=0)

    print('training data shape', x_train_both.shape)
    print('training label shape', y_train_both.shape)
    # train on target domain
    model_after = model_vanilla_NN(x_train_both, y_train_both,
                                   save_image_path, BATCH_SIZE, EPOCHS, L_RATE)
    # evaluation
    y_train_pred_source_domain = model_prediction(model_after, x_train_source_domain)
    y_train_pred_target_domain = model_prediction(model_after, x_train_target_domain)
    y_test_pred_source_domain = model_prediction(model_after, x_test_source_domain)
    y_test_pred_target_domain = model_prediction(model_after, x_test_target_domain)

    print("Evaluation on source domain training set: ")
    RMSE_train_source_domain, R2_train_source_domain = evaluate_regression_results(y_train_source_domain,
                                                                                               y_train_pred_source_domain)
    print("Evaluation on target domain training set: ")
    RMSE_train_target_domain, R2_train_target_domain = evaluate_regression_results(y_train_target_domain,
                                                                                               y_train_pred_target_domain)
    print("Evaluation on source domain test set: ")
    RMSE_test_source_domain, R2_test_source_domain = evaluate_regression_results(y_test_source_domain,
                                                                                             y_test_pred_source_domain)
    print("Evaluation on source domain test set: ")
    RMSE_test_target_domain, R2_test_target_domain = evaluate_regression_results(y_test_target_domain,
                                                                                             y_test_pred_target_domain)
    print("__________________________________")

    f = open(save_filename, "a+")
    f.write("year = %f frac = %f " % (experiment_years[-1], kk))
    f.write("%f %f %f %f %f %f %f %f" % (
    RMSE_train_source_domain, R2_train_source_domain, RMSE_train_target_domain, R2_train_target_domain,
    RMSE_test_source_domain, R2_test_source_domain, RMSE_test_target_domain, R2_test_target_domain))
    f.write("\n")
    f.close()

    save_scatter_plot = 1
    if save_scatter_plot == 1:
        # target domain
        reported = y_test_target_domain
        predicted = y_test_pred_target_domain
        RMSE = RMSE_test_target_domain
        R2 = R2_test_target_domain

        save_name = 'Target Domain DNN ' + str(experiment_years[-1]) + '_' + str(kk) + '%'
        print("save DNN scatter plot.")
        scatter_plot(reported, predicted, RMSE, R2, save_name, save_scatter_image_path)
        # source domain
        reported = y_test_source_domain
        predicted = y_test_pred_source_domain
        RMSE = RMSE_test_source_domain
        R2 = R2_test_source_domain

        save_name = 'Source Domain DNN ' + str(experiment_years[-1]) + '_' + str(kk) + '%'
        print("save DNN scatter plot.")
        scatter_plot(reported, predicted, RMSE, R2, save_name, save_scatter_image_path)
