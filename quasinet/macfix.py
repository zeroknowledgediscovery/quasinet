def macfix():
    import platform
    import os
    import quasinet
    install_path=os.path.dirname(quasinet.__file__)
    if platform.system() == "Darwin":
        import shutil
        print('copying dcor for mac')
        shutil.copy(install_path+'/bin/macnew_dcor.so',
                    install_path+'/bin/dcor.so')

        
