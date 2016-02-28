import reporters
from main.mainRFSelection import valenceCorrelationWorker

#reporter = reporters.HTMLAnalyticsReporter()
results = valenceCorrelationWorker('gini')
#reporter.genReport( [results] )