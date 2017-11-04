from exchangelib import DELEGATE, Account, Credentials, Configuration

creds = Credentials(
    username='<user_name>',
    password='<password>')

config = Configuration(service_endpoint='https://outlook.office365.com/EWS/Exchange.asmx', auth_type='basic', credentials=creds)

account = Account(
    primary_smtp_address='<email_address>',
    config=config,
    autodiscover=False,
    access_type=DELEGATE)

# Print first 100 inbox messages in reverse order
for item in account.inbox.all().order_by('-datetime_received')[:100]:
    print(item.subject, item.body, item.attachments)
