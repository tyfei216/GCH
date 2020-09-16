import configparser

config = configparser.ConfigParser()

config['classifier'] = {'dim_In':23341, 
                    'dim_Hid':1024, 
                    'dim_Out':100, 
                    'hid_Layers':1, 
                    'batch_Norm':False}

config['encoder'] = {'dim_In':23341, 
                    'dim_Hid':1024, 
                    'dim_Out':64, 
                    'hid_Layers':1, 
                    'batch_Norm':False}

config['generator'] = {'dim_Ran':128,
                    'dim_In':64,
                    'dim_Hid':1024,
                    'dim_Out':23341}

config['discriminator'] = {'dim_In':23341,
                        'dim_Hid':1024, 
                        'dim_Out':64}

config['training'] = {'epoch':80, 
                    'lr':0.0002}

def save_config(file_path):
    with open(file_path, 'w') as f:
        config.write(f)    

with open('mixed_23341.ini', 'w') as f:
    config.write(f)