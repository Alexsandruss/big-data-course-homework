import requests
from time import sleep
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel


class Bot:
    def __init__(self, token):
        self.token = token
        self.name = self.get_me()["first_name"]
        self.last_checked_update_id = 0

    def telegram_request(self, method, parameters=None, files=None):
        try:
            response = requests.post("https://api.telegram.org/bot" + self.token + "/" + method,
                                     params=parameters, files=files).json()
        except:
            response = {"ok": False}
        if response["ok"]:
            return response["result"]
        else:
            return {}

    def get_me(self):
        return self.telegram_request("getme")

    def get_updates(self, offset=None, limit=100, timeout=0, allowed_updates=None):
        params = {
            "offset": offset,
            "limit": limit,
            "timeout": timeout,
            "allowed_updates": allowed_updates
        }
        updates = self.telegram_request("getupdates", params)
        if len(updates) != 0:
            self.last_checked_update_id = updates[len(updates) - 1]["update_id"]
        return updates

    def send_message(self, chat_id, text):
        params = {
            'chat_id': chat_id,
            'text': text
        }
        return self.telegram_request("sendmessage", params)

    def get_last_messages(self):
        updates = self.get_updates(self.last_checked_update_id + 1, allowed_updates=["message"])
        messages = []
        for update in updates:
            if "message" in update.keys():
                messages.append(update["message"])
        return messages


def handle_message(msg):
    global model, movies_map
    out = {
        'text': 'Unknown request',
        'chat_id': msg["chat_id"]
    }
    if msg['text'].startswith('/forUser'):
        user = int(msg['text'].split(' ')[1])
        recs = model.recommendForUserSubset(spark.createDataFrame([(user,)], ['userId']), 10)
        recs = recs.collect()[0]['recommendations']
        out['text'] = ''
        for rec in recs:
            rec = movies_map[rec['movieId']]
            out['text'] += f'{rec}\n'
        out['text'] = out['text'][:-1]
    elif msg['text'].startswith('/forMovie'):
        item = int(msg['text'].split(' ')[1])
        recs = model.recommendForItemSubset(spark.createDataFrame([(item,)], ['movieId']), 10)
        recs = recs.collect()[0]['recommendations']
        out['text'] = ''
        for rec in recs:
            out['text'] += f'{rec["userId"]} '
        out['text'] = out['text'][:-1]
    return out

if __name__ == '__main__':
    # initialize Spark
    spark = SparkSession.builder \
        .master("yarn") \
        .appName("MovieLens") \
        .config(key='spark.submit.deployMode', value='client') \
        .config(key='spark.executor.instances', value=2) \
        .config(key='spark.executor.cores', value=4) \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir("hdfs:///checkpoints")
    # Load movies names
    with open('movies.dat', 'r') as movies_file:
        movies = movies_file.read().split('\n')[1:-1]
    movies_map = {}
    for movie in movies:
        movie = movie.split('::')
        movies_map.update(
            {int(movie[0]): f'{movie[1]} [{movie[2]}]'}
        )
    # Load ALS model
    model = ALSModel.load('/movie-lens-1m-model')
    # Create Telegram bot and listen incoming messages
    bot = Bot('TOKEN')
    while True:
        messages = bot.get_last_messages()
        for message in messages:
            try:
                incoming_message = {"text": message["text"], "chat_id": message["chat"]["id"]}
                print(" IN:", incoming_message)
                outgoing_message = handle_message(incoming_message)
                print("OUT:", outgoing_message)
                bot.send_message(outgoing_message["chat_id"], outgoing_message["text"])
            except Exception as e:
                print(e)
        sleep(1)
