
import argparse

from main import main_som_classifier, main_dgcytof, main_softmax



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the scripts for generating results.")

    # Add the file argument
    parser.add_argument('-m', type=str, help='The method for which to generate results')

    # Parse the arguments
    args = parser.parse_args()

    if args.m == 'som':
        main_som_classifier()
    elif args.m == 'dgcytof':
        main_dgcytof()
    elif args.m == 'softmax':
        main_softmax()
    else:
        print('Please specify a valid method.')
