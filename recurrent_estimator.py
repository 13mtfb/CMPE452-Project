# standard files
import tensorflow as tf
import pathlib as pl

# custom files
import stock_data

# paths to data
TRAIN_PATH = 'data/SP500_daily.csv'
TEST_PATH = 'data/SP500_daily.csv'

def main(argv):
    my_file = pl.Path(TRAIN_PATH)
    if my_file.is_file():
        # file exists
        dataset = stock_data.csv_input_fn(TRAIN_PATH,1000)
    else :
        print ("file does not exist")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
