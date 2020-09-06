from pymongo import MongoClient

# client = MongoClient("mongodb://localhost:27017")  # host uri
client = MongoClient("....for host uri....")
db = client.image_predition  # Select the database
image_details = db.imageData


def addNewImage(i_name, prediction, conf, time, url):
    image_details.insert({
        "file_name": i_name,
        "prediction": prediction,
        "confidence": conf,
        "upload_time": time,
        "url": url
    })


def getAllImages():
    data = image_details.find()
    return data
