import smtplib
import ssl

connection = smtplib.SMTP()
connection.connect('smtp.gmail.com')
connection.starttls()

f1=open('./gmail.pem', 'w+')
print >> f1, ssl.DER_cert_to_PEM_cert(connection.sock.getpeercert(binary_form=True))