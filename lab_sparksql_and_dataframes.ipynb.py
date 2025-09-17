# Databricks notebook source
# MAGIC %md
# MAGIC ###Exploratory Data Analysis with Pyspark and Spark SQL
# MAGIC The following notebook utilizes New York City taxi data from TLC Trip Record Data
# MAGIC
# MAGIC ##Instructions
# MAGIC - Load and explore nyc taxi data from january 0f 2019. The exercises can be executed using pyspark or spark sql ( a subset of the questions will be re-answered using the language not chosen for the main work).
# MAGIC - Load the zone lookup table to answer the questions about the nyc boroughs.
# MAGIC - Load nyc taxi data from January of 2025 and compare data.
# MAGIC - With any remaining time, work on the where to go from here section.
# MAGIC - Lab due date is TBD ( due dates will be updated in the readme for the class repo )

# COMMAND ----------

# Define the name of the new catalog
catalog = 'taxi_eda_db'

# define variables for the trips data
schema = 'yellow_taxi_trips'
volume = 'data'
file_name = 'yellow_tripdata_2019-01.parquet'
table_name = 'tbl_yellow_taxi_trips'
path_volume = '/Volumes/' + catalog + "/" + schema + '/' + volume
path_table =  catalog + "." + schema
download_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet'

# COMMAND ----------


# create the catalog/schema/volume
spark.sql('create catalog if not exists ' + catalog)
spark.sql('create schema if not exists ' + catalog + '.' + schema)
spark.sql('create volume if not exists ' + catalog + '.' + schema + '.' + volume)

# COMMAND ----------

# Get the data
dbutils.fs.cp(f"{download_url}", f"{path_volume}" + "/" + f"{file_name}")

# COMMAND ----------

# create the dataframe
df_trips = spark.read.parquet(f"{path_volume}/{file_name}",
  header=True,
  inferSchema=True,
  sep=",")

# COMMAND ----------


# Show the dataframe
df_trips.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Part 1
# MAGIC This section can be completed either using pyspark commands or sql commands ( There will be a section after in which a self-chosen subset of the questions are re-answered using the language not used for the main section. i.e. if pyspark is chosen for the main lab, sql should be used to repeat some of the questions. )
# MAGIC
# MAGIC - Add a column that creates a unique key to identify each record in order to answer questions about individual trips
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

df_trips = df_trips.withColumn("trip_id", monotonically_increasing_id())
df_trips.show(1)


# COMMAND ----------

# MAGIC %md
# MAGIC - Which trip has the highest passanger count
# MAGIC

# COMMAND ----------

df_trips.orderBy(df_trips.passenger_count.desc()).select("trip_id", "passenger_count").show(1)


# COMMAND ----------

# MAGIC %md
# MAGIC - What is the Average passanger count
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import avg

df_trips.select(avg("passenger_count")).show()


# COMMAND ----------

# MAGIC %md
# MAGIC - Shortest/longest trip by distance? by time?
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp, col, max, min

# Calculer la durée du trajet en secondes
df_trips = df_trips.withColumn(
    "trip_time",
    unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")
)

# Distance
df_trips.select(max("trip_distance").alias("max_distance"), 
                min("trip_distance").alias("min_distance")).show()

# Durée
df_trips.select(max("trip_time").alias("max_trip_time"), 
                min("trip_time").alias("min_trip_time")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC - highest/lowest faire amounts for a trip, what burough is associated with the each.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - busiest day/slowest single day
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import to_date, count

df_trips = df_trips.withColumn("trip_date", to_date("tpep_pickup_datetime"))

df_trips.groupBy("trip_date").agg(count("*").alias("num_trips")).orderBy("num_trips", ascending=False).show(1)  # Busiest
df_trips.groupBy("trip_date").agg(count("*").alias("num_trips")).orderBy("num_trips").show(1)  # Slowest


# COMMAND ----------

# MAGIC %md
# MAGIC - busiest/slowest time of day ( you may want to bucket these by hour or create timess such as morning, afternoon, evening, late night )
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import hour

df_trips = df_trips.withColumn("hour_of_day", hour("tpep_pickup_datetime"))

df_trips.groupBy("hour_of_day").agg(count("*").alias("num_trips")).orderBy("num_trips", ascending=False).show(1)  # Busiest
df_trips.groupBy("hour_of_day").agg(count("*").alias("num_trips")).orderBy("num_trips").show(1)  # Slowest


# COMMAND ----------

# MAGIC %md
# MAGIC - On average which day of the week is slowest/busiest
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import dayofweek

df_trips = df_trips.withColumn("weekday", dayofweek("tpep_pickup_datetime"))

df_trips.groupBy("weekday").agg(count("*").alias("num_trips")).orderBy("num_trips", ascending=False).show()  # Busiest to slowest


# COMMAND ----------

# MAGIC %md
# MAGIC - Does trip distance or num passangers affect tip amount
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import corr

df_trips.select(corr("trip_distance", "tip_amount").alias("corr_distance_tip"),
                corr("passenger_count", "tip_amount").alias("corr_passenger_tip")).show()


# COMMAND ----------

# MAGIC %md
# MAGIC - What was the highest "extra" charge and which trip
# MAGIC

# COMMAND ----------

df_trips.orderBy(df_trips.extra.desc()).select("trip_id", "extra").show(1)


# COMMAND ----------

# MAGIC %md
# MAGIC - Are there any datapoints that seem to be strange/outliers (make sure to explain your reasoning in a markdown cell)?

# COMMAND ----------

# Distance = 0 mais fare > 0
df_trips.filter((col("trip_distance") == 0) & (col("fare_amount") > 0)).show(5)

# Passager = 0
df_trips.filter(col("passenger_count") == 0).show(5)

# Durée > 24h (86400 secondes)
df_trips.filter(col("trip_time") > 86400).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Part 2
# MAGIC - Using the code for loading the first dataset as an example, load in the taxi zone lookup and answer the following questions
# MAGIC

# COMMAND ----------

df_zones = spark.read.csv("/path/to/taxi_zone_lookup.csv", header=True, inferSchema=True)

# Joindre avec df_trips pour avoir le borough de pickup et dropoff
df_trips = df_trips.join(df_zones.select(col("LocationID").alias("PULocationID"), col("Borough").alias("pickup_borough")), on="PULocationID", how="left")
df_trips = df_trips.join(df_zones.select(col("LocationID").alias("DOLocationID"), col("Borough").alias("dropoff_borough")), on="DOLocationID", how="left")



# COMMAND ----------

# MAGIC %md
# MAGIC - which borough had most pickups? dropoffs?
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - what are the busiest days of the week by borough?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC - what is the average trip distance by borough?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC - what is the average trip fare by borough?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC - load the dataset from the most recently available january, is there a change to any of the average metrics.
