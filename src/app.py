import os
from flask import Flask
from backend.pred import mod

UPLOAD_FOLDER = os.path.abspath('audio')
# UPLOAD_FOLDER = os.path.abspath('src/audio')

app = Flask(__name__)

app.register_blueprint(mod)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

