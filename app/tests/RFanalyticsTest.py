import reporters
from main.mainRFAnalytics import valenceCorrelationWorker, arousalCorrelationWorker

#reporter = reporters.HTMLAnalyticsReporter()
results = valenceCorrelationWorker()
#results = arousalCorrelationWorker()
#reporter.genReport( [results] )

