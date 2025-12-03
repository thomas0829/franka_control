# import zerorpc

# from droid.franka.robot import FrankaRobot

# if __name__ == "__main__":
#     robot_client = FrankaRobot()
#     s = zerorpc.Server(robot_client)
#     s.bind("tcp://0.0.0.0:4242")
#     s.run()

import os
import warnings
import logging

import colorlog           #  ➜  pip install colorlog
import zerorpc

from droid.franka.robot import FrankaRobot


# ---------- logging ----------------------------------------------------------
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s] [%(levelname)s]%(reset)s %(message_log_emoji)s %(white)s%(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG':    'blue',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )
)

LEVEL_EMOJIS = {
    'DEBUG': '🔍',
    'INFO': '✅',
    'WARNING': '⚠️ ',
    'ERROR': '❌',
    'CRITICAL': '🔥'
}

old_factory = logging.getLogRecordFactory()

def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.message_log_emoji = LEVEL_EMOJIS.get(record.levelname, '')
    return record

logging.setLogRecordFactory(record_factory)


log = colorlog.getLogger('franka_rpc')
log.addHandler(handler)
log.setLevel(logging.INFO)        #  DEBUG for chattier output
log.propagate = False  

#  1) Globally keep warnings quiet unless they are important.
warnings.filterwarnings('ignore')                # blanket → silence
#  2) …or, if you prefer to keep Deprecation / User warnings visible:
# warnings.filterwarnings('once')

# #  3) glfw “DISPLAY variable missing” notice (raises GLFWError subclass)
# warnings.filterwarnings('ignore', category=UserWarning, module='glfw')

#  4) Polymetis “failed to load libtorch…so” message: make the loader quiet
os.environ.setdefault('POLYMETIS_SUPPRESS_LOAD_WARNING', '1')

# ---------- optional: wrap the robot so every RPC call is logged ------------
class LoggedFranka(FrankaRobot):
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if callable(attr) and not name.startswith('_'):
            def _wrapper(*a, **kw):
                log.debug(f'RPC call ► {name}{a if a else ""}{kw if kw else ""}')
                return attr(*a, **kw)
            return _wrapper
        return attr


# ---------- main ------------------------------------------------------------
if __name__ == '__main__':
    log.info('Launching Franka RPC server on tcp://0.0.0.0:4242')
    server = zerorpc.Server(LoggedFranka())
    server.bind('tcp://0.0.0.0:4242')
    server.run()
