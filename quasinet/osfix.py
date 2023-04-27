def osfix(OS=None):
    import shutil
    import platform
    import os
    import quasinet
    install_path=os.path.dirname(quasinet.__file__)
    if OS is None:
        if platform.system() == "Darwin":
            shutil.copy(install_path+'/bin/macnew_dcor.so',
                        install_path+'/bin/dcor.so')
            shutil.copy(install_path+'/bin/macnew_Cfunc.so',
                    install_path+'/bin/Cfunc.so')
    if OS == 'macarm':
            shutil.copy(install_path+'/bin/macnew_dcor.so',
                        install_path+'/bin/dcor.so')
            shutil.copy(install_path+'/bin/macnew_Cfunc.so',
                    install_path+'/bin/Cfunc.so')
    if OS == 'win':
            shutil.copy(install_path+'/bin/win_dcor.so',
                        install_path+'/bin/dcor.so')
            shutil.copy(install_path+'/bin/win_Cfunc.so',
                    install_path+'/bin/Cfunc.so')
    if OS == 'macx86':
            shutil.copy(install_path+'/bin/macx86_dcor.so',
                        install_path+'/bin/dcor.so')
            shutil.copy(install_path+'/bin/macx86_Cfunc.so',
                    install_path+'/bin/Cfunc.so')
        
        
