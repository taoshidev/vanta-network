import os
from enum import Enum
from multiprocessing import Pool


class ParallelizationMode(Enum):
    SERIAL = 0
    PYSPARK = 1
    MULTIPROCESSING = 2

def get_multiprocessing_pool(parallel_mode: ParallelizationMode, num_processes: int = 0):
    print(f"parallel_mode: {parallel_mode} ({type(parallel_mode)})")
    pool = None
    if parallel_mode == ParallelizationMode.MULTIPROCESSING:
        print("Creating multiprocessing pool...")
        pool = Pool(num_processes) if num_processes else Pool()
        print(f"Pool created: {pool}")
    else:
        print("Not using multiprocessing mode.")
    return pool
def get_spark_session(parallel_mode: ParallelizationMode):
    if parallel_mode == ParallelizationMode.PYSPARK:
        # Check if running in Databricks
        is_databricks = 'DATABRICKS_RUNTIME_VERSION' in os.environ
        # Initialize Spark
        if is_databricks:
            # In Databricks, 'spark' is already available in the global namespace
            print("Running in Databricks environment, using existing spark session")
            should_close = False

        else:
            # Create a new Spark session if not in Databricks
            from pyspark.sql import SparkSession

            print("getOrCreate Spark session")
            spark = SparkSession.builder \
                .appName("PerfLedgerManager") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "6g") \
                .config("spark.executor.cores", "4") \
                .config("spark.driver.maxResultSize", "2g") \
                .getOrCreate()
            should_close = True

    else:
        spark = None
        should_close = False

    return spark, should_close
