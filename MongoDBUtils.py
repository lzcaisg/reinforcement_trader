import pymongo


def setUpMongoDB(db_name="testing", col_name="S&P500"):
    client = pymongo.MongoClient(
        "mongodb+srv://lzcai:raspberry@freecluster-q4nkd.gcp.mongodb.net/test?retryWrites=true&w=majority")
    # db = client['testing']
    # collection = db['S&P500']
    db = client[db_name]
    collection = db[col_name]
    return client, db, collection