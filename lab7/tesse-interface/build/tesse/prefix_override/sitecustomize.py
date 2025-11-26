import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ghabibi/vnav_ws2/src/tesse-interface/install/tesse'
