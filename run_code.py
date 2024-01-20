import argparse
import numpy as np
import warnings
import os
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, roc_auc_score
from models.DRANetSplicer import DRANetSplicer
from data.encode_data import encode_and_split_data
from visualization.deep_shap import deep_shap
from visualization.grad_cam import grad_cam_heatmap

# Train or pred results report
def result_report(y_true, y_pred):
    class_true = np.argmax(y_true,axis=1)
    class_pred = np.argmax(y_pred,axis=1)
    class_report = classification_report(class_true, class_pred, labels=[0, 1], target_names=['site', 'non_site'], output_dict=True)
    auc = roc_auc_score(y_true, y_pred)
    acc = class_report['accuracy']
    err = 1-acc
    pre = class_report['site']['precision']
    sn = class_report['site']['recall']
    sp = class_report['non_site']['recall']
    f1 = class_report['site']['f1-score']
    report = "Acc: {:.4f}, Pre: {:.4f}, Sn: {:.4f}, Sp: {:.4f}, Err: {:.4f}, F1: {:.4f}, Auc: {:.4f}".format(acc,pre,sn,sp,err,f1,auc)
    return report

# Calculation of learning rate
def temp(model):
    def step_decay(epoch):
        lrate = K.get_value(model.optimizer.lr)
        drop = 0.5
        epochs_drop = 5
        if (1 + epoch) % epochs_drop == 0:
            lrate = lrate * drop
        return lrate
    return step_decay

# Train the model
def train(args, train_dataset_path, val_dataset_path):
    # load datasets
    train_dataset = np.load(train_dataset_path)
    x_train = train_dataset['x_train']
    y_train = train_dataset['y_train']

    val_dataset = np.load(val_dataset_path)
    x_val = val_dataset['x_val']
    y_val = val_dataset['y_val']

    # input data shape
    height = x_train.shape[1]
    width = x_train.shape[2]
    if len(x_train.shape)==3:
        channels = 1
    elif len(x_train.shape)==4:
        channels = train_dataset.shape[3]
    data_shape = (height, width, channels)

    # model params
    batch_size = args.batch_size
    epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    verbose = args.verbose

    model = DRANetSplicer(data_shape)
    # model.summary()
    opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    lrate = LearningRateScheduler(temp(model))
    save_model_path = f"models/trained_models/{args.organism_name}_{args.splice_site_type}.h5"
    checkpoint = ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=verbose, mode='auto', save_freq='epoch', save_best_only=True)
    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, lrate], verbose=verbose, shuffle=True)

    # evaluate trained model
    best_model = load_model(save_model_path)
    
    evaluate_trained_model = best_model.predict(x_val, batch_size=batch_size, verbose=verbose)

    report = result_report(y_val, evaluate_trained_model)
    
    return report

# Test the model
def predict(args, test_dataset_path, trained_model_path):
    # load datasets
    test_dataset = np.load(test_dataset_path)
    x_test = test_dataset['x_test']
    y_test = test_dataset['y_test']

    # model params
    batch_size = args.batch_size
    verbose = args.verbose

    # load trained model
    model = load_model(trained_model_path)
    prediction = model.predict(x_test, batch_size=batch_size, verbose=verbose)

    report = result_report(y_test, prediction)

    return report

# Model visualization
def visualization(args, train_dataset_path, test_dataset_path, trained_model_path):
    # load datasets
    train_dataset = np.load(train_dataset_path)
    test_dataset = np.load(test_dataset_path)
    x_train = train_dataset['x_train']
    x_test = test_dataset['x_test']
    y_test = test_dataset['y_test']

    # load trained model
    model = load_model(trained_model_path)

    # selecting the background dataset and the dataset for visualization
    background_data = x_train[:100]
    x = x_test[:1000]
    y = y_test[:1000]

    title = f"{args.organism_name} {args.splice_site_type}"

    if args.visualization == 'deep_shap':
        fig_positive, fig_negative = deep_shap(x, background_data, model, title=title)
    if args.visualization == 'grad_cam':
        fig_positive, fig_negative = grad_cam_heatmap(x, y, model, layer_name='conv2d_20', title=title)

    return fig_positive, fig_negative

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--organism_name",
        default=None,
        type=str,
        required=True,
        help="The input organism name."
    )

    parser.add_argument(
        "--splice_site_type",
        default=None,
        type=str,
        choices=["donor", "acceptor"],
        required=True,
        help="Use this option to select donor or acceptor splice sites."
    )

    parser.add_argument(
        "--use_our_trained_models",
        default=None,
        type=str,
        choices=["DRANetSplicer1", "DRANetSplicer2", "DRANetSplicer3"],
        help="Use this option to select our trained model prediction data, unuse this option if you want to test your own re-trained model."
    )

    parser.add_argument("--test", action='store_true', default=False, help='Perform testing.')
    parser.add_argument("--train", action='store_true', default=False, help='Perform training and saving.')
    parser.add_argument("--visualization", default=None, type=str, choices=["deep_shap", "grad_cam"], help='Use deep_shap or grad_cam for model visualization.')
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--verbose", default=0, type=int, choices=[0, 1], help='Displays detailed information about the model during training or testing.')
    parser.add_argument("--report", action='store_true', default=False, help='Generate a report of the results for the given paradigm.')

    args = parser.parse_args()

    train_dataset_path = f"data/encode_datasets/{args.organism_name}_{args.splice_site_type}_train.npz"
    val_dataset_path = f"data/encode_datasets/{args.organism_name}_{args.splice_site_type}_val.npz"
    test_dataset_path = f"data/encode_datasets/{args.organism_name}_{args.splice_site_type}_test.npz"

    # Encode, split and save datasets
    if (
        not os.path.exists(train_dataset_path)
        or not os.path.exists(val_dataset_path)
        or not os.path.exists(test_dataset_path)
    ):
        print("***** Encoding data *****")
        splice_site_seq_path = f"data/dna_sequences/{args.organism_name}_{args.splice_site_type}_positive.txt"
        non_splice_site_seq_path = f"data/dna_sequences/{args.organism_name}_{args.splice_site_type}_negative.txt"
        x_train,x_val,x_test,y_train,y_val,y_test = encode_and_split_data(splice_site_seq_path,non_splice_site_seq_path)

        np.savez(train_dataset_path,x_train=x_train,y_train=y_train)
        np.savez(val_dataset_path,x_val=x_val,y_val=y_val)
        np.savez(test_dataset_path,x_test=x_test,y_test=y_test)
        print("***** Coded datasets have been saved to 'data/dna_sequences/' *****")

    # Train
    if args.train:
        print("***** Running train *****")
        result_report_train = train(args, train_dataset_path, val_dataset_path)
        if args.report:
            print("***** Train results *****")
            print(result_report_train)
            save_results_path = f"results/train_{args.organism_name}_{args.splice_site_type}.txt"
            with open(save_results_path, 'wt') as f:
                f.write(result_report_train)
            print(f"***** Trained results have been saved to {save_results_path} *****")

    # make sure the model used is ours or retrained.
    if args.use_our_trained_models is not None:
        trained_model_path = f"models/trained_models/{args.use_our_trained_models}/{args.organism_name}_{args.splice_site_type}.h5"
    else:
        trained_model_path = f"models/trained_models/{args.organism_name}_{args.splice_site_type}.h5"
    
    # make sure the model is exist.
    assert os.path.exists(trained_model_path), f"{trained_model_path} is not exist."

    # Test
    if args.test:
        print("***** Running pred *****")
        result_report_test = predict(args, test_dataset_path, trained_model_path)
        if args.report:
            print("***** Pred results *****")
            print(result_report_test)
            save_results_path = f"results/test_{args.organism_name}_{args.splice_site_type}.txt"
            with open(save_results_path, 'wt') as f:
                f.write(result_report_test)
            print(f"***** Pred results have been saved to {save_results_path} *****")

    # visualization
    if args.visualization is not None:
        print("***** Running visualization *****")
        fig_positive, fig_negative = visualization(args, train_dataset_path, test_dataset_path, trained_model_path)
        save_fig_path = "visualization/figure/"
        fig_positive.savefig(f"{save_fig_path}{args.visualization}_{args.organism_name}_{args.splice_site_type}_positive_wcs.png")
        fig_negative.savefig(f"{save_fig_path}{args.visualization}_{args.organism_name}_{args.splice_site_type}_negative_wcs.png")
        print(f"***** Figs have been saved to {save_fig_path} *****")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    main()
