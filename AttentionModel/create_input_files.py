from UtilsAtt import create_dataset, create_input_files


if __name__ == "__main__":
    '''
        previewData
            |__ drdName
                |__ 000000.jpg (previewImgs)
            |__ dataset_drdName.json
            |__ TRAIN_IMAGES_drdName_xx_previewTime_xx_sWindow.hdf5
            |__ TRAIN_SPEEDS_drdName_xx_previewTime_xx_sWindow.json
    '''
    dataPath = './previewData'

    drdFiles = ['../Data/std_001.csv', '../Data/std_002.csv']  # drd: driving record data
    drdFiles_te = ['../Data/std_003.csv']  # test data

    drdName = 'std001'  # recFreq = 20

    recFreq = 20  # [Hz]
    dvRate = 2  # predict every (recFreq/sWindow)Hz record points [unit:(1/recFreq)]
    previewTime = 10  # [s]

    # Save previews as imgs and corresponding speed targets
    create_dataset(dataPath, drdFiles, drdFiles_te, drdName, previewTime=previewTime, myDPI=120, recFreq=recFreq, dvRate=dvRate)

    # Create input files from dataset.json file for model training
    create_input_files(dataPath, drdName, previewTime=previewTime)
