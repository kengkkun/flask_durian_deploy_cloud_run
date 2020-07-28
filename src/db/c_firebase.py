import pyrebase


config = {
  'apiKey': "AIzaSyAQKROF-qLGbcQ4Dx8gyME2byQkis2wPNw",
  'authDomain': "durian-classification.firebaseapp.com",
  'databaseURL': "https://durian-classification.firebaseio.com",
  'storageBucket': "durian-classification.appspot.com"
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