import reporters
from main.mainRFSelection import valenceWorker

#reporter = reporters.HTMLAnalyticsReporter()
results = valenceWorker('gini')
#reporter.genReport( [results] )