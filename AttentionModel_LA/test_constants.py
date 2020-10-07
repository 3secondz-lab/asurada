import pdb
import argparse

alpha_ylimMax = 0.02  # for alpha visualization

''' python test_main.py -d YJ -v 164 -e 10 -c mugello -m 0 '''

''' == User Variable (It must be the same with the parameters used at Training) == '''
parser = argparse.ArgumentParser()
parser.add_argument('--driver', '-d')
parser.add_argument('--vNumber', '-v')
parser.add_argument('--epoch', '-e')
parser.add_argument('--circuit', '-c')
parser.add_argument('--mode', '-m')  # 1:simulation mode, 0:check learning result

args = parser.parse_args()

driver = args.driver
vNumber = args.vNumber
epoch = args.epoch

version = int(str(vNumber[0]))

dim = 32

curvatureLength = 250  #[m]
historyLength = 1  # [s]
predLength = 20  # [0.1s]
''' ============================================================================== '''

print('\nCheck your test env. If everything is okay, press "c". If not, press "q" and fix (@test_constants.py).\n')
print('Model version', version)
print('Model hidden dim', dim)
print('Preview Length [m]', curvatureLength)
print('Pred Length [0.1s]', predLength, '\n')

pdb.set_trace()
