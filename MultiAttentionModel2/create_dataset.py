
import utils

if __name__ == "__main__":

    ''' DataSet1: IJF_D* (Real) '''
    ''' Training & Testing in the SAME circuit
        --> Difficult to evaluate whether it replicates the driver's speed control
            according to the curvature of the road ahead. '''
    # driverNum = 1  # 1-5
    # dataPath = '../Data/IJF_D{}'.format(driverNum)
    # drdFiles_tr = ['{}/std_001.csv'.format(dataPath),
    #                '{}/std_002.csv'.format(dataPath),
    #                '{}/std_003.csv'.format(dataPath),
    #                '{}/std_006.csv'.format(dataPath),]
    # drdFiles_vl = ['{}/std_005.csv'.format(dataPath)]
    # drdFiles_te = ['{}/std_007.csv'.format(dataPath)]

    # recFreq = 20  # [Hz]

    # output_fname = 'IJF_D{}'.format(driverNum)

    ''' DataSet2: 003_0422_IJ (Real) '''
    ''' This dataset has driving data collected at two different circuit (IJ, YY). '''
    # dataPath = '../Data/003_0422_IJ/2_stdFile'
    # drdFiles_tr = ['{}/setup.csv'.format(dataPath)]
    # drdFiles_vl = ['{}/set2-2.csv'.format(dataPath)]
    # drdFiles_te = ['{}/set4-2.csv'.format(dataPath)]
    #
    # recFreq = 10  # [Hz]
    #
    # output_fname = 'IJ'

    '''  DataSet3: YJ (Simulated) '''
    dataPath = '../Data/YJ'
    drdFiles_tr = [('{}/mugello-recoder-scz_msgs.csv'.format(dataPath), '{}/mugello.csv'.format(dataPath))]  # (record, map)
    drdFiles_vl = [('{}/imola-recoder-scz_msgs.csv'.format(dataPath), '{}/imola.csv'.format(dataPath))]
    drdFiles_te = [('{}/magione-recoder-scz_msgs.csv'.format(dataPath), '{}/magione.csv'.format(dataPath))]

    recFreq = 10

    output_fname = 'YJ_TD'

    ''' Create datset '''
    ## save json file v1 - previewType: TIME
    # utils.create_dataset(drdFiles_tr, drdFiles_vl, drdFiles_te, recFreq,
    #                    previewType='TIME', previewTime=10.0,
    #                    output_path='../Data', output_fname=output_fname, historyTime=10.0)

    ## save json file v2 - previewType: DISTANCE

    ## save json file v3 - preview curvature from map information, not trajectories
    utils.create_dataset(drdFiles_tr, drdFiles_vl, drdFiles_te, recFreq,
                       previewTime=10.0, previewDistance=100.0,
                       output_path='../Data', output_fname=output_fname, historyTime=10.0,
                       hasCircuitInfo=True)

    ## save curvature/ targetSpeed/ historySpeed json files
    # cWindow:  window size of preview curvature [unit: m]
    # vWindow:  window size of targetSpeed [unit: sec]
    # vpWindow: window size of historySpeed [unit: sec]
    # cUnit:    resampling data with unit recFreq/cUnit  # (developing...)
    # vUnit:    resampling data with unit recFreq/vUnit  (ex. 10/10 --> 1 point/ 1 sec)
    # vpUnit:   resampling data with unit recFreq/vpUnit
    utils.create_input_files('../Data/{}.json'.format(output_fname), output_path='../Data', output_fname=output_fname,
                           cWindow=100, vWindow=2, vpWindow=2, cUnit=10, vUnit=10, vpUnit=10)

'''
# Dataset List
|__ DATA
    |__ IJF_D1
    |__ IJF_D2
    |__ ...
    |__ IJF_D5
    |__ 003_0422_IJ
    |__ 003_0420_YY
    |__ YJ
'''
