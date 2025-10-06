
import argparse

from main import main_som_classifier, main_dgcytof, main_fcnn, main_gatemeclass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the scripts for generating results.')

    methods = ['som', 'dgcytof', 'fcnn', 'gatemeclass']

    # Add the file argument
    parser.add_argument(
        '-m', '--method',
        type=str,
        required=True,
        choices=methods,
        help=f'Select the method to run. Choices: {', '.join(methods)}'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run script
    method_to_script = {
        'som': main_som_classifier,
        'dgcytof': main_dgcytof,
        'fcnn': main_fcnn,
        'gatemeclass': main_gatemeclass,
    }

    method_to_script[args.method]()
