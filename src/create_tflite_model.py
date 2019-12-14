from train import load_trained_model

#def load_trained_model(log_dir):
#    """ this function loads a train model at log_dir, whose name is "best_model.h5" """
#    model = create_model()
#
#    model.load_weights(os.path.join(log_dir, "best_model.h5"))
#    return model
    

def convert_to_tf(log_dir):
    model = load_trained_model(log_dir)


if __name__ == '__main__':
    log_dir = r"/home/datduong/gdrive/projects/P10-Object-Segmentation/logs/004"
    convert_to_tf(log_dir)
