import configparser

config_file = 'testing_settings.ini'

content = configparser.ConfigParser()
content.read(config_file)

try:
    gui = content['simulation'].getboolean('gui')
    print('GUI:', gui)
except KeyError as e:
    print(f"KeyError: Missing section or key - {e}")
