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


class LogHandler(object):
    __metaclass__ = Singleton

    def __init__(self):
        self.entries = dict()

    def timestamp(self):
        return datetime.fromtimestamp(time.time())

    def log_entry(self, entry):
        self.entries[self.timestamp()] = entry

    def save_log(self):
        file = open(variables.output_path + self.entries.keys()[0].strftime("%H_%M_%S"))
        for key, val in self.entries:
            file.write(key.strftime("%H_%M_%S") + " : " + val)
            file.write("\r\n")
        file.close()

    def send_findings(self):
        if not variables.should_send_mail:
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

