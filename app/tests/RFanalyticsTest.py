import reporters
from main.mainRFAnalytics import valenceCorrelationWorker, arousalCorrelationWorker

reporter = reporters.HTMLAnalyticsReporter()
results = valenceCorrelationWorker(1)
results = arousalCorrelationWorker(1)
reporter.genReport( [results] )

