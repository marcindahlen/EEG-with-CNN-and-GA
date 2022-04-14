import smtplib
from email.message import EmailMessage
import imghdr

from utils import variables
from datetime import datetime, time


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(object):
    __metaclass__ = Singleton

    def __init__(self):
        self.entries = dict()

    def timestamp(self):
        return datetime.fromtimestamp(time.time())

    def log_entry(self, entry):
        # TODO
        pass

    def save_log(self):
        file = open(variables.output_path + self.entries.keys()[0].strftime("%H_%M_%S"))
        file.close()
        # TODO
        pass

    def send_findings(self):
        if not variables.should_send_mail:
            return

        # TODO
        return

        msg = EmailMessage()
        msg['Subject'] = ''
        msg['To'] = ', '.join(variables.email_addresses)

        msg.set_content(self.entries_to_str())

        for file in variables.out_charts_path:
            with open(file, 'rb') as chart:
                img = chart.read()
            msg.add_attachment(img, maintype='image', subtype=imghdr.what(None,img))

        with smtplib.SMTP('localhost') as s:
            s.send_message(msg)

