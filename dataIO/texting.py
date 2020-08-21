import smtplib
from email.message import EmailMessage
import imghdr

from utils import variables


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

    def log_entry(self):
        pass

    def save_entries(self):
        pass

    def entries_to_str(self):
        return self.entries

    def plot_charts(self):
        pass

    def send_findings(self):
        if not variables.email_addresses:
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

