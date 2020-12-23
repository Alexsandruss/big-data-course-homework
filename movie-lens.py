from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from timeit import default_timer as timer

t0 = timer()

spark = SparkSession.builder \
    .master("yarn") \
    .appName("MovieLens") \
    .config(key='spark.submit.deployMode', value='client') \
    .config(key='spark.executor.instances', value=2) \
    .config(key='spark.executor.cores', value=4) \
    .getOrCreate()

sc = spark.sparkContext
sc.setCheckpointDir("hdfs:///checkpoints")

data_size = '1m'
ratings = spark.read.csv(f"/ratings-{data_size}.csv", schema="col0 INT, col1 INT, col2 DOUBLE, col3 INT") \
    .toDF("userId", "movieId", "rating", "timestamp")

als = ALS(
    rank=64,
    regParam=0.015,
    maxIter=20,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    nonnegative = True,
    implicitPrefs = False,
    coldStartStrategy="drop"
)

model = als.fit(ratings)

model.save(f'/movie-lens-{data_size}-model')

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

predictions = model.transform(ratings)
RMSE = evaluator.evaluate(predictions)
t1 = timer()
print(RMSE)
print(t1-t0)
