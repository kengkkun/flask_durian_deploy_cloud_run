import pyrebase


config = {
  'apiKey': "Your API key",
  'authDomain': "Your-Domain.firebaseapp.com",
  'databaseURL': "Your-Domain.firebaseio.com",
  'storageBucket': "Bucket-storage"
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()

def set_data(collection, macId, filename, label, conf, datatime, url):
    data = {
        u'filename': f'{filename}',
        u'prediction': f'{label}',
        u'confidence': f'{conf}',
        u'upload_time': f'{datatime}',
        u'url': f'{url}'
    }
    db.child(collection).child(macId).set(data)


def get_data(collection):
    data = db.child(f'{collection}').get()
    return data.val()
