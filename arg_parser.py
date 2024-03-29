import argparse
'''
example of a command argument assuming that current directory is my_utility/python_script
    python __init__.py --merge
'''

'''
    with default
        parser.add_argument('--dataset', type=str, default='gene_disease', help='specify type of dataset to be used')

    with action 
        parser.add_argument('--subgraph', action="store_true", help='NOT CURRENTLY COMPATIBLE WITH THE PROGRAM;Use only node in the largest connected component instead of all nodes disconnected graphs')

    with nargs, this is used to extract provided arguments as a list 
        eg --something 1 2 3 4 5 
            args.something == [1,2, 3,4,5] is true
        parser.add_argument('--weighted_class', default=[1,1,1,1,1], nargs='+', help='list of weighted_class of diseases only in order <0,1,2,3,4,5>')
'''
parser = argparse.ArgumentParser()

#--------main
# parser.add_argument('--lmda', type=str, default=0.1, help='lamda value for MSE')
parser.add_argument('--lmda', type=float, default=0.1, help='lamda value for MSE')
parser.add_argument('--lr', type=float, default=1, help='lamda value for MSE')
parser.add_argument('--bs', type=int, default=32, help='lamda value for MSE')
parser.add_argument('--epochs', type=int, default=10, help='lamda value for MSE')
parser.add_argument('--cv', type=int, default=5, help='lamda value for MSE')
parser.add_argument('--verbose', action='store_true', help="verbose")
parser.add_argument('--report_performance', action='store_true', help="verbose")

#-- utilities


args = parser.parse_args()
