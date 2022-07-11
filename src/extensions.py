#Firebase
import firebase_admin
from firebase_admin import firestore, storage
credentials = firebase_admin.credentials.Certificate("./firebase-service-account-key.json")
firebase_admin.initialize_app(credentials)
database = firestore.client()
cloud_storage = firebase_admin.storage.bucket("icmp-d3412.appspot.com")
