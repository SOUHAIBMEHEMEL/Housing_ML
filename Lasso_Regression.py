#import dataset
from pandas import np

from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.sql import SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)

data = sqlContext.read.csv(
    "C:\\Users\\asmaa lagrid\\Desktop\\S2\\BDM\\TP_BDM\\Housing_ML\\housing_data.csv", inferSchema=True,
    header=False)
data.show(10)

# add features column
from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=["_c0","_c1","_c2","_c3","_c4","_c5","_c6","_c7","_c8","_c9",
                                            "_c10","_c11","_c12","_c13"],outputCol="features")
output=featureassembler.transform(data)
output.show()

# get our variables and our collected results
mv = output.select("_c13")
finalized_data=output.select("features","_c13")
finalized_data = finalized_data.withColumnRenamed("_c13", "label")
finalized_data.show()

# Lasso regression
from pyspark.ml.regression import LinearRegression
print("Lasso Regression bound 0.01")

lasso = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.01,
                         elasticNetParam=1.0)
lasso_model = lasso.fit(finalized_data)
print("Coefficients: " + str(lasso_model.coefficients))
print("Intercept: " + str(lasso_model.intercept))

# get our variables and our collected results
finalized_data=output.select("features","_c13")
finalized_data = finalized_data.withColumnRenamed("_c13", "label")
TrainSet,TestSet=finalized_data.randomSplit([0.8,0.2])
TrainSet.show()
TestSet.show()
# get gcv min
lr_predictions = lasso_model.transform(TestSet)
lr_predictions = lr_predictions.select("label","features")
lr_predictions.show(5)
# Run cross-validation, and choose the best set of parameters.
paramGrid = ParamGridBuilder()\
    .addGrid(lasso.regParam, [0.01]) \
    .addGrid(lasso.fitIntercept, [False, True])\
    .addGrid(lasso.elasticNetParam, [1.0])\
    .build()
# In this case the estimator is Lasso the linear regression.
crossval = CrossValidator(estimator=lasso,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3)

newModel = crossval.fit(TrainSet)
evaluation = RegressionEvaluator(predictionCol="prediction",labelCol="label",metricName="rmse")
evaluation.evaluate(newModel.transform(lr_predictions))

# Lasso opt
print("Lasso Regression bound 0.8")
lasso_opt = LinearRegression(featuresCol = 'Features', labelCol='_c13', maxIter=10, regParam=0.8,
                             elasticNetParam=1.0)
lasso_opt_model = lasso.fit(finalized_data)
print("Coefficients: " + str(lasso_model.coefficients))
print("Intercept: " + str(lasso_model.intercept))

#get error
trainingSummary = lasso_opt_model.summary
print("MSE: %f" % trainingSummary.meanSquaredError)
print("R2: %f" % trainingSummary.r2)


