from json import dumps
from server.db import c_firebase

from flask import Blueprint

api = Blueprint('api', __name__, template_folder='templates')

api.route('/')
def api():
    return dumps(c_firebase.get_data('sounds'))
